import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F
import diffdist
import torch.distributed as dist


def gather(z):
    gather_z = [torch.zeros_like(z) for _ in range(torch.distributed.get_world_size())]
    gather_z = diffdist.functional.all_gather(gather_z, z)
    gather_z = torch.cat(gather_z)

    return gather_z


def accuracy(logits, labels, k):
    topk = torch.sort(logits.topk(k, dim=1)[1], 1)[0]
    labels = torch.sort(labels, 1)[0]
    acc = (topk == labels).all(1).float()
    return acc


def mean_cumulative_gain(logits, labels, k):
    topk = torch.sort(logits.topk(k, dim=1)[1], 1)[0]
    labels = torch.sort(labels, 1)[0]
    mcg = (topk == labels).float().mean(1)
    return mcg


def mean_average_precision(logits, labels, k):
    # TODO: not the fastest solution but looks fine
    argsort = torch.argsort(logits, dim=1, descending=True)
    labels_to_sorted_idx = torch.sort(torch.gather(torch.argsort(argsort, dim=1), 1, labels), dim=1)[0] + 1
    precision = (1 + torch.arange(k, device=logits.device).float()) / labels_to_sorted_idx
    return precision.sum(1) / k


class NTXent(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """
    LARGE_NUMBER = 1e9

    def __init__(self, tau=1., gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.

    def forward(self, z, get_map=False):
        n = z.shape[0]
        assert n % self.multiplier == 0

        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        logits = z @ z.t()
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        logprob = F.log_softmax(logits, dim=1)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n//m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        # TODO: maybe different terms for each process should only be computed here...
        loss = -logprob[np.repeat(np.arange(n), m-1), labels].sum() / n / (m-1) / self.norm

        # zero the probability of identical pairs
        pred = logprob.data.clone()
        pred[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER
        acc = accuracy(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)

        if get_map:
            _map = mean_average_precision(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)
            return loss, acc, _map

        return loss, acc

class NeuralEFLoss(nn.Module):
    LARGE_NUMBER = 1e9

    def __init__(self, batch_size, device, multiplier=2, distributed=False, riemannian_projection=False):
        super().__init__()
        self.batch_size = batch_size
        self.multiplier = multiplier # how many times we repeat each batch
        self.distributed = distributed
        self.riemannian_projection = riemannian_projection

        if self.distributed:
            batch_size *= dist.get_world_size()

        '''
        when multiplier=2, the sim_matrix is like
                x_1  x_2  x_3  x_4 x'_1 x'_2 x'_3 x'_4
        x_1  |    1    0    0    0    1    0    0    0
        x_2  |    0    1    0    0    0    1    0    0
        x_3  |    0    0    1    0    0    0    1    0
        x_4  |    0    0    0    1    0    0    0    1
        x'_1 |    1    0    0    0    1    0    0    0
        x'_2 |    0    1    0    0    0    1    0    0
        x'_3 |    0    0    1    0    0    0    1    0
        x'_4 |    0    0    0    1    0    0    0    1
        '''
        A = torch.tile(torch.eye(batch_size), (multiplier, multiplier)).to(device)

        print("the similarity matrix is like:")
        print(torch.tile(torch.eye(4), (multiplier, multiplier)))

        D = A.sum(-1).diag()
        L = D - A
        L_normalized = D.inverse().sqrt() @ L @ D.inverse().sqrt()

        self.K = L_normalized


    def forward(self, z, get_map=False):
        n, k = z.shape[0], z.shape[1]
        assert n == self.batch_size * self.multiplier

        # just collect the features from multiple workers
        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        # ensure that each eigenfunction is normalized
        psi_x = z / z.norm(dim=0).clamp(min=1e-6) * math.sqrt(n)

        # estimate the neuralef grad
        with torch.no_grad():
            K_psi = self.K @ psi_x
            psi_K_psi = psi_x.T @ K_psi
            mask = torch.eye(k, device=z.device) - \
                (psi_K_psi / psi_K_psi.diag().clamp(min=1e-6)).tril(diagonal=-1).T
            grad = K_psi @ mask

            # the trick in eigengame paper
            if self.riemannian_projection:
                grad.sub_((psi_x*grad).sum(0) * psi_x / n)

            # the scaling may be unnecessary
            grad *= - 1 # 2 / n**2

        # it is a pseudo loss whose gradient w.r.t. psi_x is the `grad'
        pseudo_loss = (psi_x * grad).sum()

        # measure the accuracy of the instance discrimination task
        with torch.no_grad():
            logits = z @ z.t()
            logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER
            logits = F.log_softmax(logits, dim=1)

            # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
            m = self.multiplier
            labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n//m, n)) % n
            # remove labels pointet to itself, i.e. (i, i)
            labels = labels.reshape(n, m)[:, 1:].reshape(-1)

            acc = accuracy(logits, torch.LongTensor(labels.reshape(n, m-1)).to(logits.device), m-1)

            if get_map:
                _map = mean_average_precision(logits, torch.LongTensor(labels.reshape(n, m-1)).to(logits.device), m-1)
                return pseudo_loss, acc, _map

        return pseudo_loss, acc
