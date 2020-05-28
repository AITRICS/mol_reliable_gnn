import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

#from layers.Bayes_By_BackProp_layer import BayesLinear_Normalq
#from layers.Bayes_By_BackProp_LR_layer import BayesLinear_local_reparam as BayesLinear_Normalq
#from layers.prior import laplace_prior, isotropic_gauss_prior

#from blitz.modules import BayesianLinear
from layers.linear_bayesian_layer import BayesianLinear
#from layers.linear_bayesian_LR_layer import BayesianLinear
from blitz.utils import variational_estimator

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""

class GINLayer(nn.Module):
    """
    [!] code adapted from dgl implementation of GINConv

    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula.
    aggr_type :
        Aggregator type to use (``sum``, ``max`` or ``mean``).
    out_dim :
        Rquired for batch norm layer; should match out_dim of apply_func if not None.
    dropout :
        Required for dropout of output features.
    graph_norm : 
        boolean flag for output features normalization w.r.t. graph sizes.
    batch_norm :
        boolean flag for batch_norm layer.
    init_eps : optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter.
    
    """
    def __init__(self, apply_func, aggr_type, dropout, graph_norm, batch_norm, layer_norm, init_eps=0.0, learn_eps=False):
        super().__init__()
        self.apply_func = apply_func
        
        if aggr_type == 'sum':
            self._reducer = fn.sum
        elif aggr_type == 'max':
            self._reducer = fn.max
        elif aggr_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggr_type))
            
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.dropout = dropout
        
        in_dim = apply_func.mlp.input_dim
        out_dim = apply_func.mlp.output_dim
            
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))
            
        if self.batch_norm:
            self.bn_node_h = nn.BatchNorm1d(out_dim)
        if self.layer_norm:
            self.ln_node_h = nn.LayerNorm(out_dim)
        
    def forward(self, g, h, snorm_n):

        g = g.local_var()
        g.ndata['h'] = h
        g.update_all(fn.copy_u('h', 'm'), self._reducer('m', 'neigh'))
        h = (1 + self.eps) * h + g.ndata['neigh']

        if self.apply_func is not None:
            h = self.apply_func(h)

        if self.graph_norm:
            h = h* snorm_n # normalize activation w.r.t. graph size
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  

        if self.layer_norm:
            h = self.ln_node_h(h) # layer normalization  
        
        h = F.relu(h) # non-linear activation
        
        h = F.dropout(h, self.dropout, training=self.training)
        return h
    
    
class ApplyNodeFunc(nn.Module):
    """
        This class is used in class GINNet
        Update the node feature hv with MLP
    """
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, h):
        
        h = self.mlp(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, input_dim, hidden_dim, output_dim, batch_norm, layer_norm):

        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        # Multi-layer model
        self.linear_1 = BayesianLinear(hidden_dim, hidden_dim, bias=False)
        if self.batch_norm:
            self.bn = nn.BatchNorm1d((hidden_dim))
        if self.layer_norm:
            self.ln = nn.LayerNorm(hidden_dim)
        self.linear_2 = BayesianLinear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        h = self.linear_1(x)
        if self.batch_norm:
            h = self.bn(h)
        if self.layer_norm:
            h = self.ln(h)
        h = F.relu(h)

        return self.linear_2(h)
