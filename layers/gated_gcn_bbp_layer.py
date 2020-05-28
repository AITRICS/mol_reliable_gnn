import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.linear_bayesian_layer import BayesianLinear

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""

class GatedGCNLayer(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, graph_norm, batch_norm, layer_norm, gated_gcn_agg):#, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.gated_gcn_agg = gated_gcn_agg
        
        self.A = BayesianLinear(input_dim, output_dim, bias=False)
        self.B = BayesianLinear(input_dim, output_dim, bias=False)
        self.C = BayesianLinear(input_dim, output_dim, bias=False)
        self.D = BayesianLinear(input_dim, output_dim, bias=False)
        self.E = BayesianLinear(input_dim, output_dim, bias=False)

        if batch_norm:
            self.bn_node_h = nn.BatchNorm1d(output_dim)
            self.bn_node_e = nn.BatchNorm1d(output_dim)
        if layer_norm:
            self.ln_node_h = nn.LayerNorm(output_dim)
            self.ln_node_e = nn.LayerNorm(output_dim)

    def message_func(self, edges):
        Bh_j = edges.src['Bh']    
        e_ij = edges.data['Ce'] +  edges.src['Dh'] + edges.dst['Eh'] # e_ij = Ce_ij + Dhi + Ehj
        #   e_ij -> C is C, D is A, E is B
        e_ij = F.relu(e_ij)
        edges.data['e'] = e_ij
        return {'Bh_j' : Bh_j, 'e_ij' : e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij'] 
        sigma_ij = torch.sigmoid(e) # sigma_ij = sigmoid(e_ij)

        if self.gated_gcn_agg == "mean":
            h = Ah_i + torch.mean( sigma_ij * Bh_j, dim=1 ) # hi = Ahi + mean_j alpha_ij * Bhj 
        elif self.gated_gcn_agg == "sum":
            h = Ah_i + torch.sum( sigma_ij * Bh_j, dim=1 ) / ( torch.sum( sigma_ij, dim=1 ) + 1e-6 )  # hi = Ahi + sum_j eta_ij/sum_j' eta_ij' * Bhj <= dense attention       
        else:
            print("Undefined Aggregation")

        return {'h' : h}
    
    def forward(self, g, h, e, snorm_n, snorm_e):
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        g.edata['e']  = e 
        g.edata['Ce'] = self.C(e) 
        g.update_all(self.message_func,self.reduce_func) 
        h = g.ndata['h'] # result of graph convolution
        e = g.edata['e'] # result of graph convolution
        
        if self.graph_norm:
            h = h* snorm_n # normalize activation w.r.t. graph size
            e = e* snorm_e # normalize activation w.r.t. graph size
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  
            e = self.bn_node_e(e) # batch normalization  

        if self.layer_norm:
            h = self.ln_node_h(h) # layer normalization
            e = self.ln_node_e(e) # layer normalization
        
        h = F.relu(h) # non-linear activation
        e = F.relu(e) # non-linear activation
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)
