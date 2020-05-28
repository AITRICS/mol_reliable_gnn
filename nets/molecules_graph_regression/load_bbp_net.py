"""
    Utility file to select Bayes by Backprop GraphNN model as
    selected by the user
"""

from nets.molecules_graph_regression.gin_bbp_net import GINNet
from nets.molecules_graph_regression.gcn_bbp_net import GCNNet
from nets.molecules_graph_regression.gated_gcn_bbp_net import GatedGCNNet
from nets.molecules_graph_regression.graphsage_bbp_net import GraphSageNet
from nets.molecules_graph_regression.gat_bbp_net import GATNet

def GCN(net_params):
    return GCNNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GraphSage(net_params):
    return GraphSageNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GCN': GCN,
        'GatedGCN': GatedGCN,
        'GraphSage': GraphSage,
        'GAT': GAT,
        'GIN': GIN
    }
        
    return models[MODEL_NAME](net_params)
