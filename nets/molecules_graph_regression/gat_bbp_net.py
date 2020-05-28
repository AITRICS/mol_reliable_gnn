import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from layers.gat_bbp_layer import GATLayer
from layers.mlp_readout_bbp_layer import MLPReadout

from layers.linear_bayesian_layer import BayesianLinear


class GATNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.layer_norm = net_params['layer_norm']
        self.residual = net_params['residual']

        self.att_reduce_fn = net_params['att_reduce_fn']

        self.task = net_params['task']
        if self.task == 'classification':
            self.num_classes = net_params['num_classes']
        
        self.dropout = dropout
        
        self.embedding_lin = BayesianLinear(num_atom_type, hidden_dim, bias=False)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([
            GATLayer(hidden_dim, hidden_dim, num_heads, dropout, 
                     self.graph_norm, self.batch_norm, self.layer_norm, self.att_reduce_fn) for _ in range(n_layers)])

        self.linear_ro = BayesianLinear(hidden_dim, out_dim, bias=False)        
        self.linear_predict = BayesianLinear(out_dim, 1, bias=True)

		#	additional parameters for gated gcn
        if self.residual == "gated":
            self.W_g  = nn.Linear(2*hidden_dim, hidden_dim, False)
        
    def forward(self, g, h, e, snorm_n, snorm_e):

        #   modified dtype for new dataset
        h = h.float()

        h = self.embedding_lin(h)
        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h_in = h
            h = conv(g, h, snorm_n)
            if self.residual:
                if self.residual == "gated":
                    z = torch.sigmoid(self.W_g(torch.cat([h, h_in], dim=1)))
                    h = z * h + (torch.ones_like(z) - z)*h_in
                else:
                    h += h_in	

        g.ndata['h'] = self.linear_ro(h)
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.sum_nodes(g, 'h')  # default readout is summation
            
        return self.linear_predict(hg)
    
    def loss(self, scores, targets):
        if self.task == 'regression':
            loss = nn.MSELoss()(scores, targets)
        else:
            loss = nn.BCEWithLogitsLoss()(scores, targets)
        return loss
       
