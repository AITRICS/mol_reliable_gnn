import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from layers.linear_bayesian_layer import BayesianLinear

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
    
# Sends a message of node feature h
# Equivalent to => return {'m': edges.src['h']}
msg = fn.copy_src(src='h', out='m')

def reduce_mean(nodes):
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}

def reduce_sum(nodes):
    accum = torch.sum(nodes.mailbox['m'], 1)
    return {'h': accum}

class NodeApplyModule(nn.Module):
    # Update node feature h_v with (Wh_v+b)
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = BayesianLinear(in_dim, out_dim, bias=False)
        
    def forward(self, node):
        h = self.linear(node.data['h'])
        return {'h': h}

class GCNLayer(nn.Module):
    """
        Param: [in_dim, out_dim]
    """
    def __init__(self, in_dim, out_dim, activation, dropout, graph_norm, batch_norm, layer_norm, agg, residual=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.residual = residual
        self.agg = agg
        
        if in_dim != out_dim:
            self.residual = False
        
        self.apply_mod = NodeApplyModule(in_dim, out_dim)
        if batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_dim)
        if layer_norm:
            self.layernorm_h = nn.LayerNorm(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, g, feature, snorm_n):

        h_in = feature   # to be used for residual connection
        g.ndata['h'] = feature
        if self.agg == "mean":
            g.update_all(msg, reduce_mean)
        elif self.agg == "sum":
            g.update_all(msg, reduce_sum)
        else:
            g.update_all(msg, reduce_sum)

        g.apply_nodes(func=self.apply_mod)
        h = g.ndata['h'] # result of graph convolution
        
        if self.graph_norm:
            h = h * snorm_n # normalize activation w.r.t. graph size
        if self.batch_norm:
            h = self.batchnorm_h(h) # batch normalization  
        if self.layer_norm:
            h = self.layernorm_h(h) # layer normalization  
        
        h = self.activation(h)
            
        h = self.dropout(h)
        return h
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.residual)
