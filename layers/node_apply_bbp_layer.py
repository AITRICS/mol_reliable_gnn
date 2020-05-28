"""

! Code started from dgl diffpool examples dir
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.linear_bayesian_layer import BayesianLinear

class NodeApply(nn.Module):
    """
    Works -> the node_apply function in DGL paradigm
    """
    def __init__(self, in_feats, out_feats, activation, dropout, concat_norm, bias=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = BayesianLinear(in_feats * 2, out_feats, bias)
        self.activation = activation
        self.concat_norm = concat_norm

    def concat(self, h, aggre_result):
        bundle = torch.cat((h, aggre_result), 1)
        bundle = self.linear(bundle)
        return bundle

    def forward(self, node):
        h = node.data['h']
        c = node.data['c']
        bundle = self.concat(h, c)
        
        if self.activation:
            bundle = self.activation(bundle)
        if self.concat_norm == True:
            bundle = F.normalize(bundle, p=2, dim=1)
        return {"h": bundle}
