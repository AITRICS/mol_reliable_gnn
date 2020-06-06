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
    def __init__(self, in_feats, out_feats, activation, dropout, concat_norm, bias=False,
            prior_sigma_1=0.1, prior_sigma_2=0.001, prior_pi=1.):
        super().__init__()

        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi

        self.dropout = nn.Dropout(p=dropout)
        self.linear = BayesianLinear(in_feats * 2, out_feats, bias,
                prior_sigma_1=self.prior_sigma_1,
                prior_sigma_2=self.prior_sigma_2,
                prior_pi=self.prior_pi)
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
