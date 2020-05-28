import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Analyzing Learned Molecular Representations for Property Prediction
    https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237

    Discriminative Embeddings of Latent Variable Models for Structured Data
    https://arxiv.org/abs/1603.05629
"""

class D_MPNNLayer(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, graph_norm, batch_norm, layer_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False

        if batch_norm:
            self.bn_node_h = nn.BatchNorm1d(output_dim)
        if layer_norm:
            self.ln_node_h = nn.LayerNorm(output_dim)

        self.f_e1 = nn.Linear(3*input_dim, input_dim, bias=False)
        self.f_e2 = nn.Linear(input_dim, output_dim, bias=False)

    def get_neighbor_edge_func(self, edges):
        #   output: concatenated x_k and h_kv

        h_kv = edges.data['h']
        x_k = edges.src['x']

        return {'x_k_h_kv': torch.cat([x_k, h_kv], dim=-1)}

    def message_func(self, nodes):

        x_k_h_kv = nodes.mailbox['x_k_h_kv']
        x_v = nodes.data['x'].unsqueeze(1).repeat(1, x_k_h_kv.shape[1], 1)
        m = F.relu(self.f_e1(torch.cat([x_v, x_k_h_kv], dim=-1)))
        m = torch.sum(m, dim=1, keepdims=True)
        
        return {'m': m}

    def reduce_func(self, edges):
        h = edges.data['h']
        m = edges.src['m'].squeeze(1)
        #   version 1: lin + residual connection
        # h = F.relu(self.f_e2(m))
        #   version 2: no resid connection
        h = self.f_e2(m) 
        # might need to be skip connected with h_0? or just h_t-1?
        #   might be overlapped with residual connections outside each layer

        return {'h': h}

    
    def forward(self, g, x, h, snorm_n, snorm_e):

        g.ndata['x'] = x
        g.edata['h'] = h
            

        g.update_all(self.get_neighbor_edge_func, self.message_func)
        g.apply_edges(self.reduce_func)
        h = g.edata['h']
        h = h.squeeze(0)
        
        if self.graph_norm:
            h = h* snorm_n # normalize activation w.r.t. graph size
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  

        if self.layer_norm:
            h = self.ln_node_h(h) # layer normalization  
        
        h = F.relu(h) # non-linear activation
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h
    
