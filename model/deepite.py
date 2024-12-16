import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool
from model.gumbel_softmax import gumbel_softmax

class DeepITE(nn.Module):
    def __init__(self, args):
        super(DeepITE, self).__init__()
        self.logit_pai = nn.Parameter(torch.logit(torch.Tensor([args.pai])))

        # Encoder layers
        self.gnn_repre_layer1 = GATConv(args.input_dim, args.hidden1_dim)
        self.gnn_repre_layer2 = GATConv(args.hidden1_dim, args.hidden1_dim)

        self.node_repre_layer1 = GATConv(args.hidden1_dim, args.hidden1_dim)
        self.node_repre_layer2 = GATConv(args.hidden1_dim, args.hidden1_dim)
        self.node_repre_linear = nn.Linear(args.hidden1_dim, 2, bias=True)

        self.edge_repre_layer1 = GATConv(args.hidden1_dim, args.hidden1_dim)
        self.edge_repre_layer2 = GATConv(args.hidden1_dim, args.hidden1_dim)
        self.edge_repre_linear = nn.Linear(args.hidden1_dim, 1, bias=True)

        self.graph_repre_layer1 = GATConv(args.hidden1_dim, args.hidden2_dim)
        self.graph_repre_layer2 = GATConv(args.hidden2_dim, args.hidden2_dim)
        self.graph_repre_linear = nn.Linear(args.hidden2_dim, 2, bias=True)

        # Decoder layers
        self.decoder_layer1 = nn.Linear(args.input_dim, args.hidden1_dim, bias=True)
        self.decoder_layer2 = nn.Linear(args.hidden1_dim, args.input_dim, bias=True)

        if not args.model_path:
            self.init_weights()
        self.args = args

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)

    def encode(self, X, adj, tau):
        # GNN representation
        hidden = self.gnn_repre_layer1(x=X, edge_index=adj)
        hidden = self.gnn_repre_layer2(x=hidden, edge_index=adj)

        # Node representation
        node_u = self.node_repre_layer1(x=hidden, edge_index=adj)
        node_u = self.node_repre_layer2(x=node_u, edge_index=adj)
        node_u = self.node_repre_linear(node_u)
        u_mean = node_u[..., 0].view(-1, 1)
        u_logstd = node_u[..., 1].view(-1, 1)
        sampled_u = []
        for _ in range(self.args.sample_num):
            sampled_u.append(torch.randn_like(u_mean) * torch.exp(0.5 * u_logstd) + u_mean)
        sampled_u = torch.stack(sampled_u, dim=0)
        # sampled_u = torch.randn_like(u_mean) * torch.exp(0.5 * u_logstd) + u_mean

        # Edge representation
        edge_logit_W = self.edge_repre_layer1(x=hidden, edge_index=adj)
        edge_logit_W = self.edge_repre_layer2(x=edge_logit_W, edge_index=adj)
        edge_logit_W = self.edge_repre_linear(edge_logit_W)
        Y = gumbel_softmax(torch.sigmoid(edge_logit_W), tau, X.size(0), sample_num=self.args.sample_num).view(self.args.sample_num, -1, X.size(0))

        # Graph representation
        Z = self.graph_repre_layer1(x=hidden, edge_index=adj)
        Z = self.graph_repre_layer2(x=Z, edge_index=adj)
        Z = self.graph_repre_linear(Z)
        Z_maxpool = global_max_pool(Z, None, None)
        z_mean = Z_maxpool[0, 0]
        z_logstd = Z_maxpool[0, 1]

        return sampled_u, edge_logit_W, Y, z_mean, z_logstd, u_mean, u_logstd

    def decode(self, sampled_u, Y, adj_direct):
        w_adj_direct = torch.mul(Y, adj_direct)
        x_recon = []
        for d_i in range(sampled_u.shape[0]):
            cur_w_adj_direct = w_adj_direct[d_i]
            weighted_adj_direct = torch.inverse(torch.eye(cur_w_adj_direct.shape[0]) - cur_w_adj_direct.transpose(0, 1))
            deco_input = torch.matmul(weighted_adj_direct, self.decoder_layer1(sampled_u[d_i]))
            x_recon.append(self.decoder_layer2(deco_input))
        x_recon = torch.cat(x_recon).view(self.args.sample_num, -1, self.args.input_dim)
        return x_recon

    def forward(self, X, adj, adj_direct, tau):
        sampled_u, logit_W, Y, z_mean, z_logstd, u_mean, u_logstd = self.encode(X, adj, tau)
        x_recon = self.decode(sampled_u, Y, adj_direct)
        return x_recon, logit_W, z_mean, z_logstd, u_mean, u_logstd, self.logit_pai