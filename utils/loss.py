import torch

def ELBO(X, X_recon, logit_W, z_mean, z_logstd, u_mean, u_logstd, logit_pai, adj_w):
    x_shape = X.shape[0]

    ELBO_u = - 0.5 * torch.sum(u_mean ** 2 + torch.exp(u_logstd) - u_logstd - 1)

    sigmoid_logit_W = torch.sigmoid(logit_W)
    sigmoid_logit_pai = torch.sigmoid(logit_pai)

    ELBO_Y = torch.mean(torch.sum(sigmoid_logit_W * adj_w * logit_pai, dim=1)) - \
             torch.mean(torch.sum(sigmoid_logit_W * adj_w * logit_W, dim=1)) + \
             torch.sum(adj_w * (torch.log(sigmoid_logit_pai) - logit_pai)) - \
             torch.mean(torch.sum(adj_w * (torch.log(sigmoid_logit_W) - logit_W), dim=1))
    ELBO_Y = ELBO_Y / torch.sum(adj_w)

    ELBO_z = 0.5 * z_mean + z_logstd / 2

    recon_diff = X - X_recon
    ELBO_x_tem = torch.sum(recon_diff ** 2, dim=2)
    ELBO_x = - torch.exp(z_mean + torch.exp(z_logstd) / 2) / 2 * torch.sum(torch.mean(ELBO_x_tem, dim=0)) / x_shape

    ELBO_total = ELBO_u + ELBO_Y + ELBO_z + ELBO_x

    return - ELBO_total

def ELBO_LOGMLE(X, X_recon, logit_W, z_mean, z_logstd, u_mean, u_logstd, logit_pai, adj_w, Y_true):
    x_shape = X.shape[0]

    ELBO_u = - 0.5 * torch.sum(u_mean ** 2 + torch.exp(u_logstd) - u_logstd - 1)

    sigmoid_logit_W = torch.sigmoid(logit_W)
    LOGMLE_Y = torch.sum(adj_w * Y_true * torch.log(1 - sigmoid_logit_W)) + \
               torch.sum(adj_w * (1 - Y_true) * torch.log(sigmoid_logit_W))
    LOGMLE_Y = LOGMLE_Y / torch.sum(adj_w)

    ELBO_z = 0.5 * z_mean + z_logstd / 2

    recon_diff = X - X_recon
    ELBO_x_tem = torch.sum(recon_diff ** 2, dim=2)
    ELBO_x = - 0.5 * torch.exp(z_mean + torch.exp(z_logstd) / 2) * torch.sum(torch.mean(ELBO_x_tem, dim=0)) / x_shape

    ELBO_LOGMLE_total = ELBO_u + LOGMLE_Y + ELBO_z + ELBO_x

    return - ELBO_LOGMLE_total