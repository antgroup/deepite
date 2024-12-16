from utils.loss import ELBO, ELBO_LOGMLE
import torch

def get_XYA(batch, device):
    X = batch['X'].unsqueeze(-1).squeeze(0).to(device)
    Y = batch['Y'].unsqueeze(-1).squeeze(0).to(device)
    adj = batch['adj'].squeeze(0).to(device)  # directed
    return X, Y, adj

def calculate_tau(step):
    if step <= 50:
        tau = 101 - 2 * step
    else:
        tau = 0.5 / (step - 50)
    return tau

def calculate_loss(X, Y, x_recon, logit_W, z_mean, z_logstd, u_mean, u_logstd, logit_pai, adj_w, supervised_rate):
    supervised_mark = torch.rand(1).item()
    if supervised_mark < supervised_rate:
        loss = ELBO_LOGMLE(
            X, x_recon, logit_W, z_mean, z_logstd, u_mean, u_logstd, logit_pai, adj_w, Y
        )
    else:
        loss = ELBO(
            X, x_recon, logit_W, z_mean, z_logstd, u_mean, u_logstd, logit_pai, adj_w
        )

    return loss