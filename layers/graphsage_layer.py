import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""

from layers.sage_aggregator_layer import SumAggregator, MaxPoolAggregator, MeanAggregator, LSTMAggregator
from layers.node_apply_layer import NodeApply

class GraphSageLayer(nn.Module):

    def __init__(self, in_feats, out_feats, activation, dropout,
                 aggregator_type, graph_norm, batch_norm, layer_norm, concat_norm, bias=False):
        super().__init__()
        self.in_channels = in_feats
        self.out_channels = out_feats
        self.aggregator_type = aggregator_type
        self.concat_norm = concat_norm
        
        self.nodeapply = NodeApply(in_feats, out_feats, activation, dropout,
                               concat_norm,
                               bias=False)
        self.dropout = nn.Dropout(p=dropout)

        if aggregator_type == "maxpool":
            self.aggregator = MaxPoolAggregator(in_feats, in_feats,
                                                activation, bias)
        elif aggregator_type == "lstm":
            self.aggregator = LSTMAggregator(in_feats, in_feats)
        elif aggregator_type == "sum":
            self.aggregator = SumAggregator()
        else:
            self.aggregator = MeanAggregator()
        
        if batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_feats)
        if layer_norm:
            self.layernorm_h = nn.LayerNorm(out_feats)

        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

    def forward(self, g, h, snorm_n=None):
        h = self.dropout(h)
        g.ndata['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'), self.aggregator,
                     self.nodeapply)
        h = g.ndata['h']
        
        if self.graph_norm: 
            h = h * snorm_n
        if self.batch_norm:
            h = self.batchnorm_h(h)

        return h
    
