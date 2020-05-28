import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Learning Deep Generative Models of Graphs 
    https://arxiv.org/pdf/1711.07553v2.pdf
"""

class GGNNLayer(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, graph_norm, batch_norm, layer_norm):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        if batch_norm:
            self.bn_node_h = nn.BatchNorm1d(output_dim)
        else:
            self.bn_node_h = None

        if layer_norm:
            self.ln_node_h = nn.LayerNorm(output_dim)
        else:
            self.ln_node_h = None
        
        #   edge embed with one hot, 2 layer mlp
        self.f_e1 = nn.Linear(2*input_dim+6, 2*input_dim+6, bias=False)
        self.f_e2 = nn.Linear(2*input_dim+6, output_dim, bias=False)

        self.GRU = nn.GRU(output_dim, output_dim, bias=False)


    def message_func(self, edges):
        h0_e = torch.cat([edges.src['h'], edges.data['e']], dim=1) 
        return {'m': h0_e}

    def reduce_func(self, nodes):
        h1 = nodes.data['h']
        m = nodes.mailbox['m']
        h1_h0_e = torch.cat([h1.unsqueeze(1).expand(-1, m.size(1), -1), m], dim=2)

        a_v = F.relu(self.f_e1(h1_h0_e))
        a_v = self.f_e2(a_v)
        a_v = torch.sum(a_v, dim=1, keepdims=True)

        return {'h': a_v}

    def forward(self, g, h, e, snorm_n, snorm_e):
        
        g.ndata['h'] = h
        g.edata['e'] = e
            

        g.update_all(self.message_func,self.reduce_func) 
        a_v = g.ndata['h']
        
        _, h = self.GRU(a_v.view(1, -1, self.out_channels), h.unsqueeze(0))
        h = h.squeeze(0)
        
        if self.graph_norm:
            h = h* snorm_n # normalize activation w.r.t. graph size
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  

        if self.layer_norm:
            h = self.ln_node_h(h) # layer normalization  
        
        h = F.relu(h) # non-linear activation
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)
