import math
import os
import sys
import time

import numpy
import pandas as pd
import torch
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt

from model.deepite import DeepITE

from data.data import load_data

from utils.utils import get_XYA
from utils.arguments import prepare_args
from utils.loss import ELBO, ELBO_LOGMLE
from utils.preprocessing import adj_to_coo, directed_to_undirected

from utils.metrics import MMD, SSE, recall_k

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def write_tensorboard(summary_writer: SummaryWriter, log_dict: dict, completed_steps):
    for key, value in log_dict.items():
        summary_writer.add_scalar(f'{key}', value, completed_steps)

def test_Recall():
    args = prepare_args()

    test_dataset = load_data(args.dataset, args.valid_dataset_path)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
    )

    model = DeepITE(args).to(device)
    if args.model_path:
        model = model.load_state_dict(args.model_path).to(device)

    accumulated_steps = 0
    accumulated_recall_1 = 0
    accumulated_recall_3 = 0
    accumulated_recall_5 = 0
    tau = 0.001
    model.eval()
    for step, batch in enumerate(test_dataloader):
        X, Y, adj = get_XYA(batch, device)
        adj_coo = adj_to_coo(directed_to_undirected(adj))
        x_recon, logit_W, z_mean, z_logstd, u_mean, u_logstd, logit_pai = model(X, adj_coo, adj, tau)
        recall_1 = recall_k(logit_W, Y, k=1)
        recall_3 = recall_k(logit_W, Y, k=3)
        recall_5 = recall_k(logit_W, Y, k=5)
        accumulated_recall_1 += recall_1
        accumulated_recall_3 += recall_3
        accumulated_recall_5 += recall_5
        accumulated_steps += 1

    recall_1 = accumulated_recall_1 / accumulated_steps
    recall_3 = accumulated_recall_3 / accumulated_steps
    recall_5 = accumulated_recall_5 / accumulated_steps
    print(f"Recall@1: {recall_1}")
    print(f"Recall@3: {recall_3}")
    print(f"Recall@5: {recall_5}")

def test_MMD():
    args = prepare_args()

    test_dataset = load_data(args.dataset, args.valid_dataset_path)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
    )

    model = DeepITE(args).to(device)
    if args.model_path:
        model = model.load_state_dict(args.model_path).to(device)

    accumulated_steps = 0
    accumulated_MMD = 0.0
    tau = 0.001
    model.eval()
    for step, batch in enumerate(test_dataloader):
        X, Y, adj = get_XYA(batch, device)
        adj_coo = adj_to_coo(directed_to_undirected(adj))
        x_recon, logit_W, z_mean, z_logstd, u_mean, u_logstd, logit_pai = model(X, adj_coo, adj, tau)
        mmd = MMD(X, x_recon)
        accumulated_MMD += mmd
        accumulated_steps += 1

    mmd = accumulated_MMD / accumulated_steps
    print(f"MMD: {mmd}")

def test_SSE():
    args = prepare_args()

    test_dataset = load_data(args.dataset, args.valid_dataset_path)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
    )

    model = DeepITE(args).to(device)
    if args.model_path:
        model = model.load_state_dict(args.model_path).to(device)

    accumulated_steps = 0
    accumulated_SSE = 0.0
    tau = 0.001
    model.eval()
    for step, batch in enumerate(test_dataloader):
        X, Y, adj = get_XYA(batch, device)
        adj_coo = adj_to_coo(directed_to_undirected(adj))
        x_recon, logit_W, z_mean, z_logstd, u_mean, u_logstd, logit_pai = model(X, adj_coo, adj, tau)
        sse = SSE(X, x_recon)
        accumulated_SSE += sse
        accumulated_steps += 1

    sse = accumulated_SSE / accumulated_steps
    print(f"SSE: {sse}")