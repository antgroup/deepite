import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

def sample_gumbel(shape, eps=1e-20, sample_num=100, device='cpu'):
    new_shape = [sample_num]
    for shape_i in shape:
        new_shape.append(shape_i)
    U = torch.rand(new_shape).to(device)
    # if args.cuda:
    #     U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, sample_num=100):
    y = logits + sample_gumbel(logits.size(), sample_num=sample_num, device=logits.device)
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, out_dim, hard=False, sample_num=100):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature, sample_num)

    if not hard:
        return y.view(-1, out_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, out_dim)

