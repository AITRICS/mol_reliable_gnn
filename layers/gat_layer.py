import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""

class GATHeadLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout, graph_norm, batch_norm, layer_norm, att_reduce_fn="softmax"):
        super().__init__()
        out_dim = out_dim // num_heads
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        if batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_dim)
        if layer_norm:
            self.layernorm_h = nn.LayerNorm(out_dim)

        self.att_reduce_fn = att_reduce_fn

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        #   other attention mechanism such as cosine dist can be implemented
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        if self.att_reduce_fn == "softmax":
            alpha = F.softmax(nodes.mailbox['e'], dim=-1)
        elif self.att_reduce_fn == "tanh":
            alpha = torch.tanh(nodes.mailbox['e'])
        else:
            print("Undefined Aggregation")

        alpha = F.dropout(alpha, self.dropout, training=self.training)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h, snorm_n):
        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']
        if self.graph_norm:
            h = h * snorm_n
        if self.batch_norm:
            h = self.batchnorm_h(h)
        if self.layer_norm:
            h = self.layernorm_h(h) # layer normalization  

        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class GATLayer(nn.Module):
    """
        Param: [in_dim, out_dim, n_heads]
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout, graph_norm, batch_norm, layer_norm, residual=False, att_reduce_fn="softmax"):
        super().__init__()
        self.in_channels = in_dim 
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.residual = residual
        self.att_reduce_fn = att_reduce_fn
        
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATHeadLayer(in_dim, out_dim, num_heads, dropout, graph_norm, batch_norm, layer_norm, att_reduce_fn))
        self.linear_concat = nn.Linear(out_dim, out_dim, bias=False)
        self.merge = 'cat' 

    def forward(self, g, h, snorm_n):
        # h_in = h # for residual connection
        head_outs = [attn_head(g, h, snorm_n) for attn_head in self.heads]
        if self.merge == 'cat':
            #   default dim set is 128 -> 128//4 * 4heads -> mlp -> 128
            h = torch.cat(head_outs, dim=1)
            h = self.linear_concat(h)
        else:
            h = torch.mean(torch.stack(head_outs))
        
        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)
