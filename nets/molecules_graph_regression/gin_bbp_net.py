import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""

from layers.gin_bbp_layer import GINLayer, ApplyNodeFunc, MLP

from layers.linear_bayesian_layer import BayesianLinear
from blitz.utils import variational_estimator

class GINNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        dropout = net_params['dropout']
        self.n_layers = net_params['L']
        n_mlp_layers = net_params['n_mlp_GIN']               # GIN
        learn_eps = net_params['learn_eps_GIN']              # GIN
        neighbor_aggr_type = net_params['neighbor_aggr_GIN'] # GIN
        self.readout = net_params['readout']                      # this is graph_pooling_type
        self.graph_norm = net_params['graph_norm']      
        self.batch_norm = net_params['batch_norm']
        self.layer_norm = net_params['layer_norm']
        self.residual = net_params['residual']

        self.task = net_params['task']
        if self.task == 'classification':
            self.num_classes = net_params['num_classes']
        else:
            self.num_classes = 1

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        
        self.embedding_lin = BayesianLinear(num_atom_type, hidden_dim, bias=False)
        
        for layer in range(self.n_layers):
            mlp = MLP(hidden_dim, hidden_dim, hidden_dim, self.batch_norm, self.layer_norm)
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, self.graph_norm, self.batch_norm, self.layer_norm, 0, learn_eps))

        # Linear function for graph poolings (readout) of output of each layer
        # which maps the output of different layers into a prediction score

        self.linear_ro = BayesianLinear(hidden_dim, out_dim, bias=False)
        self.linear_prediction = BayesianLinear(out_dim, self.num_classes, bias=True)
        
		#	additional parameters for gated residual connection
        if self.residual == 'gated':
            self.W_g = BayesianLinear(2*hidden_dim, hidden_dim, bias=False)

    def forward(self, g, h, e, snorm_n, snorm_e):
        
        #   modified dtype for new dataset
        h = h.float()

        h = self.embedding_lin(h.cuda())
        h_in = h # for residual connection

        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h, snorm_n)

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
            hg = dgl.sum_nodes(g, 'h')  # default readout is summation

        score = self.linear_prediction(hg)

        return score

    def loss(self, scores, targets):
        if self.task == 'regression':
            loss = nn.MSELoss()(scores, targets)
        else:
            loss = nn.BCEWithLogitsLoss()(scores, targets)
        return loss
