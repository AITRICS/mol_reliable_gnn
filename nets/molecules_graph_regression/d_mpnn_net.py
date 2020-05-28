import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Analyzing Learned Molecular Representations for Property Prediction
    https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237

    Discriminative Embeddings of Latent Variable Models for Structured Data
    https://arxiv.org/abs/1603.05629
"""
from layers.d_mpnn_layer import D_MPNNLayer
from layers.mlp_readout_layer import MLPReadout

class D_MPNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.layer_norm = net_params['layer_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']

        self.task = net_params['task']
        if self.task == 'classification':
            self.num_classes = net_params['num_classes']
        
        self.embedding_h_lin = nn.Linear(num_atom_type, hidden_dim, bias=False)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.init_h_mlp = nn.Linear(hidden_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            D_MPNNLayer(hidden_dim, hidden_dim, dropout, self.graph_norm, 
                          self.batch_norm, self.layer_norm, self.residual) for _ in range(n_layers)]) 
        self.linear_ro = nn.Linear(hidden_dim, out_dim, bias=False)        
        self.linear_predict = nn.Linear(out_dim, 1, bias=True)

		#	additional parameters for gated gcn
        if self.residual == "gated":
            self.W_g  = nn.Linear(2*hidden_dim, hidden_dim, False)

        #   additional params for init h_0 on edges
        self.init_linear = nn.Linear(hidden_dim+6, hidden_dim)

    def init_message_func(self, edges):
        h_vw_0 = F.relu(self.init_linear(torch.cat([edges.src['h'], edges.data['e']], dim=1)))
        return {'h_0': h_vw_0}

    def out_reduce_func(self, nodes):
        h = nodes.mailbox['h_m']
        return {'h_out': torch.sum(h, dim=1)}
        
    def forward(self, g, h, e, snorm_n, snorm_e):
        #   modified dtype for new dataset
        h = h.float()
        e = e.float()

        # input embedding
        h = self.embedding_h_lin(h)
        h = self.in_feat_dropout(h)
        if not self.edge_feat: # edge feature set to 1
            e = torch.zeros(e.size(0),1).to(self.device)

        #   initial h embedding
        x = h # node embeds set to x
        g.ndata['h'] = h
        g.edata['e'] = e
        g.apply_edges(self.init_message_func)
        h = g.edata['h_0'] # h are defined on edges
        
        # convnets
        for num_l, conv in enumerate(self.layers):
            h_in = h
            h = conv(g, x, h, snorm_n, snorm_e)
            if self.residual:
                if self.residual == "gated":
                    z = torch.sigmoid(self.W_g(torch.cat([h, h_in], dim=1)))
                    h = z * h + (torch.ones_like(z) - z)*h_in
                else:
                    h += h_in	
        
        g.edata['h'] = h
        g.update_all(dgl.function.copy_edge(edge='h', out='h_m'), self.out_reduce_func)
        h = g.ndata['h_out']

        #   hidden states on edges are aggregated on node
        g.ndata['h'] = self.linear_ro(h)
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.sum_nodes(g, 'h')  # default readout is summation
            
        return self.linear_predict(hg)

        
    def loss(self, scores, targets):
        if self.task == 'regression':
            loss = nn.MSELoss()(scores, targets)
        else:
            loss = nn.BCEWithLogitsLoss()(scores, targets)
        return loss
