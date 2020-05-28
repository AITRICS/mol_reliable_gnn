"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import time
import random

import torch

#   Set manual seed

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    dgl.random.seed(seed)
    if torch.cuda.is_available():
       torch.cuda.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
    
# torch.set_num_threads(1)

"""
    IMPORTING CUSTOM MODULES/METHODS
"""

from nets.molecules_graph_regression.load_net import gnn_model # import all GNNS
from data.data import load_data # import dataset
from utils import get_configs, get_arguments, add_dir_name # import arguments and configurations

"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    model = model.float()
    total_param = 0
    print("MODEL DETAILS:\n")
    print(model)
    for param in model.parameters():
        print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""

def main():    
    """
        USER CONTROLS
    """
    args = get_arguments()

    args, config, params, net_params = get_configs(args)

    #   define which pipeline to be used
    if params['swa'] == True:
        from pipeline_swa import train_val_pipeline_classification
    elif params['swag'] == True:
        from pipeline_swag import train_val_pipeline_classification
    elif (params['sgld'] == True) or (params['psgld'] == True):
        from pipeline_sgld import train_val_pipeline_classification
    else:
        if params['bbp'] == True:
            from nets.molecules_graph_regression.load_bbp_net import gnn_model # import all GNNS
        from pipeline import train_val_pipeline_classification


    DATASET_NAME = config['dataset']
    MODEL_NAME = config['model']

    # setting seeds
    set_seed(params['seed'])
    print("Seed Number of Models: "+str(params['seed']))
    print("Data Seed Number: "+str(params['data_seed']))


    dataset = load_data(DATASET_NAME, args.num_train, args.num_val, args.num_test, args.data_seed, params)

    # network parameters

    #   add task information for net_params loss
    net_params['task'] = 'classification'
    net_params['num_classes'] = dataset.num_classes
    net_params['num_atom_type'] = dataset.num_atom_type
    net_params['num_bond_type'] = dataset.num_bond_type
    out_dir = config['out_dir']

    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_output_dir = out_dir + 'outputs/outputs_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')

    dirs = root_ckpt_dir, write_file_name, root_output_dir
    dirs = add_dir_name(dirs, MODEL_NAME, config, params, net_params)

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    train_val_pipeline_classification(MODEL_NAME, DATASET_NAME, dataset, config, params, net_params, dirs)
    
main()    

