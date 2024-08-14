import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.Drop = nn.Dropout(dropout)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj=None):
        Wh = torch.matmul(h, self.W) # h.shape: (batch ,N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        ###mask attention
        ####attention = torch.where(adj > 0, e, zero_vec)
        attention = e
        attention = F.softmax(attention, dim=-1)
        attention = self.Drop(attention)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return h_prime
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        B, N, E = Wh.shape  # (B, N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :]) # (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])# (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)
        # broadcast add
        e = Wh1 + Wh2.permute(0, 2, 1)  # (B, N, N)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        #self.out_att = GraphAttentionLayer(nhid * nheads, 1, dropout=dropout, alpha=alpha, concat=False)
        self.out_att = nn.Linear(nhid * nheads, 1)

        self.Drop1 = nn.Dropout(dropout)
        self.Drop2 = nn.Dropout(dropout)

    def forward(self, x, adj=None):
        ####x shape

        #x = self.Drop1(x)

        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = self.Drop2(x)
        out = self.out_att(x)
        x = self.out_att(x).squeeze(-1)

        ##return shape
        return F.softmax(x, dim=-1)


# in_features = 64
# out_features = 1
# dropout = 0.5
# alpha = 0.1
# concat = True
# nodes = torch.rand((16,64, in_features))
#
# gat = GAT(in_features, 8, 0.5, 0.2, 8)
# gat.train()
#
# out = gat(nodes)  ###batch nodes
# print(out.shape)
# print(out)
#
# Gat_layer = GraphAttentionLayer(in_features, out_features, dropout, alpha, concat)
# nodes = torch.rand((16,64, in_features))
#
# Gat_layer.train()
#
# out = Gat_layer(nodes)
# print(out.shape)
# print(out)
# out = Gat_layer(nodes)
# print(out.shape)
# print(out)
#
# Gat_layer.eval()
#
# out = Gat_layer(nodes)
# print(out.shape)
# print(out)
#
# out = Gat_layer(nodes)
# print(out.shape)
# print(out)



