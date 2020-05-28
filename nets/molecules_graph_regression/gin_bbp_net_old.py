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

# from layers.gin_layer import GINLayer, ApplyNodeFunc, MLP
from layers.gin_bbp_layer import GINLayer, ApplyNodeFunc, MLP

from layers.Bayes_By_BackProp_layer import BayesLinear_Normalq
# from layers.Bayes_By_BackProp_LR_layer import BayesLinear_local_reparam as BayesLinear_Normalq
from layers.prior import laplace_prior, isotropic_gauss_prior 

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

        # self.prior = laplace_prior(mu=0, b=1.0)
        self.prior = isotropic_gauss_prior(mu=0, sigma=0.05) #    for BBP
        # self.prior = 0.01 #    for BBP_LR

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        
        # self.embedding_lin = nn.Linear(num_atom_type, hidden_dim, False)
        self.embedding_lin = BayesLinear_Normalq(num_atom_type, hidden_dim, self.prior, False)
        
        for layer in range(self.n_layers):
            mlp = MLP(hidden_dim, hidden_dim, hidden_dim, self.batch_norm, self.layer_norm)
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           # dropout, self.graph_norm, self.batch_norm, self.layer_norm, self.residual, 0, learn_eps))
                                           dropout, self.graph_norm, self.batch_norm, self.layer_norm, 0, learn_eps))

        # Linear function for graph poolings (readout) of output of each layer
        # which maps the output of different layers into a prediction score

        # self.linear_ro = nn.Linear(hidden_dim, out_dim, bias=False)        
        # self.linear_prediction = nn.Linear(out_dim, self.num_classes, bias=True) 
        self.linear_ro = BayesLinear_Normalq(hidden_dim,  out_dim, self.prior, bias=False) 
        self.linear_prediction = BayesLinear_Normalq(out_dim, self.num_classes, self.prior, bias=True) 
        
		#	additional parameters for gated residual connection
        if self.residual == 'gated':
            self.W_g  = nn.Linear(2*hidden_dim, hidden_dim, False)

    def forward(self, g, h, e, snorm_n, snorm_e, sample=False):
        tlqw = 0
        tlpw = 0

        #   modified dtype for new dataset
        h = h.float()

        # h = self.embedding_lin(h)
        h, lqw, lpw = self.embedding_lin(h, sample)
        tlqw += lqw
        tlpw += lpw
        
        h_in = h # for residual connection

        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.n_layers):
            # h = self.ginlayers[i](g, h, snorm_n)
            h, lqw, lpw = self.ginlayers[i](g, h, snorm_n, sample)
            tlqw += lqw
            tlpw += lpw

            #pooled_h = self.pool(g, h)
            #hidden_rep.append(pooled_h)

            # Residual Connection
            if self.residual:
                if self.residual == "gated":
                    z = torch.sigmoid(self.W_g(torch.cat([h, h_in], dim=1)))
                    h = z * h + (torch.ones_like(z) - z)*h_in
                else:
                    h += h_in	
        
        # g.ndata['h'] = self.linear_ro(h)
        g.ndata['h'], lqw, lpw = self.linear_ro(h, sample)
        tlqw += lqw
        tlpw += lpw
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.sum_nodes(g, 'h')  # default readout is summation

        score, lqw, lpw = self.linear_prediction(hg, sample)
        tlqw += lqw
        tlpw += lpw

        return score, tlqw, tlpw

    def loss(self, scores, targets):
        if self.task == 'regression':
            loss = nn.MSELoss()(scores, targets)
        else:
            loss = nn.BCEWithLogitsLoss()(scores, targets)
        return loss
