import random

import torch
import torch.nn as nn
from torch.distributions import Normal

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])

    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)  #
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma = None, num_samples=None):
    batch_size = int(source.size()[0])
    if num_samples is not None:
        if batch_size > num_samples:
            idx = random.choices(list(range(batch_size)), k=num_samples)
            source = source[idx]
            target = target[idx]
            batch_size = int(source.size()[0])
    if len(source.size()) > 2:
        source = source.view(batch_size, -1)
    if len(target.size()) > 2:
        target = target.view(batch_size, -1)
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss

def SSE(source, target):
    return torch.std((source - target) ** 2)

def recall_k(preds, targets, k, one_hot_preds=False, dim=-1):
    if one_hot_preds:
        preds = preds.argmax(dim=dim)
    _, topk_indices = preds.topk(k, dim=dim, largest=True, sorted=True)
    target_ones_positions = torch.nonzero(targets == 1, as_tuple=True)[0]
    all_ones_in_topk = all(pos in topk_indices for pos in target_ones_positions)
    return 1.0 if all_ones_in_topk else 0.0

def accuracy(preds, targets, one_hot_preds=False, dim=-1):
    if one_hot_preds:
        preds = preds.argmax(dim=dim)
    return (preds == targets).sum().float() / len(targets)

def kl_divergence_normal(mu1, log_std1, mu2=None, log_std2=None):
    q = Normal(mu1, torch.exp(log_std1))
    if mu2 == None or log_std2 == None:
        mu2 = torch.zeros_like(mu1)
        log_std2 = torch.zeros_like(log_std1)
    p = Normal(mu2, torch.exp(log_std2))

    return torch.distributions.kl_divergence(q, p)