"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.molecules_graph_regression.gated_gcn_net import GatedGCNNet
from nets.molecules_graph_regression.gcn_net import GCNNet
from nets.molecules_graph_regression.gat_net import GATNet
from nets.molecules_graph_regression.graphsage_net import GraphSageNet
from nets.molecules_graph_regression.gin_net import GINNet
from nets.molecules_graph_regression.ggnn_net import GGNNet
from nets.molecules_graph_regression.d_mpnn_net import D_MPNNet


def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def GraphSage(net_params):
    return GraphSageNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def GGNN(net_params):
    return GGNNet(net_params)

def D_MPNN(net_params):
    return D_MPNNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        'GIN': GIN,
        'GGNN': GGNNet,
        'D_MPNN': D_MPNNet
    }
        
    return models[MODEL_NAME](net_params)
