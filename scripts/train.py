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
from torch.optim.nadam import NAdam
from adabelief_pytorch import AdaBelief

from data.data import load_data

from utils.arguments import prepare_args
from utils.utils import calculate_tau, calculate_loss, get_XYA
from utils.preprocessing import adj_to_coo, directed_to_undirected, generate_adj_w

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def train():
    args = prepare_args()

    print(f"Loading training dataset from {args.train_dataset_path}")
    train_dataset = load_data(args.dataset, args.train_dataset_path)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True
    )

    if args.valid_dataset_path:
        print(f"Loading validation dataset from {args.valid_dataset_path}")
        valid_dataset = load_data(args.dataset, args.valid_dataset_path)
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            pin_memory=True
        )
    else:
        valid_dataloader = None

    model = DeepITE(args).to(device)
    if args.model_path:
        print(f"Loading model from {args.model_path}")
        model = model.load_state_dict(args.model_path).to(device)
    else:
        print("Initializing model")

    # optimizer = NAdam(model.parameters(), lr=lr)
    print("Initializing optimizer and scheduler")
    optimizer = AdaBelief(model.parameters(), lr=args.learning_rate, eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

    tensorboard_dir = "tensorboard/"
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    summary_writer = SummaryWriter(log_dir=tensorboard_dir)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    supervised_rate = args.supervised_rate
    accumulated_steps = 0

    for epoch in range(args.num_epoch):
        print(f"Starting epoch {epoch + 1}/{args.num_epoch}")
        model.train()
        epoch_loss = 0

        for step, batch in enumerate(train_dataloader):
            X, Y, adj = get_XYA(batch, device)
            if X.dim() > 2 or Y.dim() > 2:
                for idx in range(X.shape[0]):
                    Xi = X[idx]
                    Yi = Y[idx]
                    adj_coo = adj_to_coo(directed_to_undirected(adj))
                    adj_w = generate_adj_w(adj)
                    tau = calculate_tau(accumulated_steps)
                    x_recon, logit_W, z_mean, z_logstd, u_mean, u_logstd, logit_pai = model(Xi, adj_coo, adj, tau)

                    loss = calculate_loss(
                        X=Xi,
                        Y=Yi,
                        x_recon=x_recon,
                        logit_W=logit_W,
                        z_mean=z_mean,
                        z_logstd=z_logstd,
                        u_mean=u_mean,
                        u_logstd=u_logstd,
                        logit_pai=logit_pai,
                        adj_w=adj_w,
                        supervised_rate=supervised_rate
                    )

                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    accumulated_steps += 1
                    epoch_loss += loss.item()

                    if accumulated_steps % args.log_interval == 0:
                        print(f"Step {accumulated_steps}: Training loss = {loss.item()}")
                        monitor(summary_writer, {'train_loss': loss.item()}, accumulated_steps)

                    if args.step_evaluation and accumulated_steps % args.evaluation_steps == 0:
                        evaluate(model, valid_dataloader, summary_writer, supervised_rate, accumulated_steps)

                    if args.step_checkpointing and accumulated_steps % args.checkpointing_steps == 0:
                        checkpoint_dir = os.path.join(args.save_path, f"{epoch}_{accumulated_steps}")
                        print(f"Step {accumulated_steps}: Saving checkpoint to {checkpoint_dir}")
                        save(model, checkpoint_dir)

            else:
                adj_coo = adj_to_coo(directed_to_undirected(adj))
                adj_w = generate_adj_w(adj)
                tau = calculate_tau(accumulated_steps)
                x_recon, logit_W, z_mean, z_logstd, u_mean, u_logstd, logit_pai = model(X, adj_coo, adj, tau)

                loss = calculate_loss(
                    X=X,
                    Y=Y,
                    x_recon=x_recon,
                    logit_W=logit_W,
                    z_mean=z_mean,
                    z_logstd=z_logstd,
                    u_mean=u_mean,
                    u_logstd=u_logstd,
                    logit_pai=logit_pai,
                    adj_w=adj_w,
                    supervised_rate=supervised_rate
                )

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accumulated_steps += 1
                epoch_loss += loss.item()

                if accumulated_steps % args.log_interval == 0:
                    print(f"Step {accumulated_steps}: Training loss = {loss.item()}")
                    monitor(summary_writer, {'train_loss': loss.item()}, accumulated_steps)

                if args.step_evaluation and accumulated_steps % args.evaluation_steps == 0:
                    evaluate(model, valid_dataloader, summary_writer, supervised_rate, accumulated_steps)

                if args.step_checkpointing and accumulated_steps % args.checkpointing_steps == 0:
                    checkpoint_dir = os.path.join(args.save_path, f"{epoch}_{accumulated_steps}")
                    print(f"Step {accumulated_steps}: Saving checkpoint to {checkpoint_dir}")
                    save(model, checkpoint_dir)


        if args.epoch_evaluation:
            evaluate(model, valid_dataloader, summary_writer, supervised_rate, accumulated_steps)

        if args.epoch_checkpointing:
            checkpoint_dir = os.path.join(args.save_path, f"{epoch}_{accumulated_steps}")
            print(f"Epoch {epoch + 1}: Saving checkpoint to {checkpoint_dir}")
            save(model, checkpoint_dir)

        monitor(summary_writer, {'epoch_loss': epoch_loss / len(train_dataloader)}, accumulated_steps)

def evaluate(model, dataloader, summary_writer, supervised_rate, step):
    if dataloader:
        print(f"Step {step}: Evaluating model")
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                X, Y, adj = get_XYA(batch, device)
                adj_coo = adj_to_coo(directed_to_undirected(adj))
                adj_w = generate_adj_w(adj)
                tau = calculate_tau(step)
                x_recon, logit_W, z_mean, z_logstd, u_mean, u_logstd, logit_pai = model(X, adj_coo, adj, tau)

                loss = calculate_loss(
                    X=X,
                    Y=Y,
                    x_recon=x_recon,
                    logit_W=logit_W,
                    z_mean=z_mean,
                    z_logstd=z_logstd,
                    u_mean=u_mean,
                    u_logstd=u_logstd,
                    logit_pai=logit_pai,
                    adj_w=adj_w,
                    supervised_rate=supervised_rate
                )

                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Evaluation completed. Average validation loss: {avg_loss}")
        monitor(summary_writer, {'valid_loss': avg_loss}, step)
        model.train()

def monitor(summary_writer, log_dict, step):
    for key, value in log_dict.items():
        summary_writer.add_scalar(f'{key}', value, step)

def save(model, checkpoint_dir):
    checkpoint_path = f"{checkpoint_dir}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved at {checkpoint_path}")

if __name__ == "__main__":
    train()