import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers.gated_gcn_layer import GatedGCNLayer
from layers.mlp_readout_layer import MLPReadout

class GatedGCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.layer_norm = net_params['layer_norm']
        self.gated_gcn_agg = net_params['gated_gcn_agg']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        
        self.task = net_params['task']
        if self.task == 'classification':
            self.num_classes = net_params['num_classes']

        self.embedding_h_lin = nn.Linear(num_atom_type, hidden_dim, bias=False)

        if self.edge_feat:
            self.embedding_e_lin = nn.Linear(num_bond_type, hidden_dim, bias=False)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim, bias=False)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([
            GatedGCNLayer(hidden_dim, hidden_dim, dropout, self.graph_norm, 
                          self.batch_norm, self.layer_norm, self.gated_gcn_agg) for _ in range(n_layers)]) 
        self.linear_ro = nn.Linear(hidden_dim, out_dim, bias=False)        
        self.linear_predict = nn.Linear(out_dim, 1, bias=True)

		#	additional parameters for gated gcn
        if self.residual == "gated":
            self.W_g  = nn.Linear(2*hidden_dim, hidden_dim, False)

        
    def forward(self, g, h, e, snorm_n, snorm_e):
        #   modified dtype for new dataset
        h = h.float()
        e = e.float()

        # input embedding
        h = self.embedding_h_lin(h)
        h = self.in_feat_dropout(h)
        if not self.edge_feat: # edge feature set to 1
            e = torch.zeros(e.size(0),1).to(self.device)

        if self.edge_feat:
            e = self.embedding_e_lin(e)
        else:
            e = self.embedding_e(e)
        
        # convnets
        for conv in self.layers:
            h_in = h
            h, e = conv(g, h, e, snorm_n, snorm_e)
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
