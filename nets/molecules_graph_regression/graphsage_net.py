import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""

from layers.graphsage_layer import GraphSageLayer
from layers.mlp_readout_layer import MLPReadout

class GraphSageNet(nn.Module):
    """
    Grahpsage network with multiple GraphSageLayer layers
    """
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        aggregator_type = net_params['sage_aggregator']
        n_layers = net_params['L']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.layer_norm = net_params['layer_norm']
        self.readout = net_params['readout']
        self.residual = net_params['residual']
        self.concat_norm = net_params['concat_norm']
        
        self.task = net_params['task']
        if self.task == 'classification':
            self.num_classes = net_params['num_classes']
        else:
            self.num_classes = 1


        self.embedding_lin = nn.Linear(num_atom_type, hidden_dim, bias=False)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu,
                                              dropout, aggregator_type,  self.batch_norm, self.graph_norm, self.layer_norm, self.concat_norm) for _ in range(n_layers)])
        self.linear_ro = nn.Linear(hidden_dim, out_dim, bias=False)
        self.linear_predict = nn.Linear(out_dim, self.num_classes, bias=True)

		#	additional parameters for gated residual connection
        if self.residual == "gated":
            self.W_g  = nn.Linear(2*hidden_dim, hidden_dim, False)
        
    def forward(self, g, h, e, snorm_n, snorm_e):
        h = h.float()

        h = self.embedding_lin(h)
        h = self.in_feat_dropout(h)
        for conv in self.layers:

            h_in = h
            h = conv(g, h, snorm_n)

            # Residual Connection
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
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        return self.linear_predict(hg)
    
    def loss(self, scores, targets):
        if self.task == 'regression':
            loss = nn.MSELoss()(scores, targets)
        else:
            loss = nn.BCEWithLogitsLoss()(scores, targets)
        return loss
