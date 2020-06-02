import subprocess
import os
import glob
import json
import torch
import argparse
import numpy as np


"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")

    return device

def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--data_seed', help="Please give a value for a random seed splitting dataset")
    parser.add_argument('--num_train', help="Please give a value for a number of training samples")
    parser.add_argument('--num_val', help="Please give a value for a number of validation samples")
    parser.add_argument('--num_test', help="Please give a value for a number of test samples")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")    
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--graph_norm', help="Please give a value for graph_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")

    parser.add_argument('--atom_meta', help="Please give a bool value whether to use atom meta info or not")
    parser.add_argument('--scaffold_split', help="Please give a bool value whether to use scaffold split or not")
    parser.add_argument('--scheduler', help="Please give a str value for which scheduler to be used")
    parser.add_argument('--step_size', help="Please give an int value for how much step size to be used for scheduler")
    parser.add_argument('--step_gamma', help="Please give a float value for how much step gamma to be used for scheduler")
    parser.add_argument('--layer_norm', help="Please give a float value for how much step gamma to be used for scheduler")

    parser.add_argument('--optimizer', default='ADAM', help="Please give a str which optimizer to be used (ADAM / SGD at the moment)")
    parser.add_argument('--grad_clip', default=0., help="Please give a float indicating how much grad (l2) norm clipped")

    #   Additional arguments for GCN
    parser.add_argument('--agg', help="Please give a value for agg in gcn")
    #   Additional arguments for GAT
    parser.add_argument('--att_reduce_fn', help="Please give a value for agg in gcn")
    #   Additional arguments for SAGE 
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--concat_norm', help="Please give a value for concat_norm")
    #   Additional arguments for GIN
    parser.add_argument('--neighbor_aggr_GIN', help="Please give a value for neighbor_aggr_GIN")
    #   Additional arguments for GatedGCN
    parser.add_argument('--gated_gcn_agg', help="Please give a value for agg in gcn")

    #   additional arguments for pretraining
    
    parser.add_argument('--save_params', help="Please give a bool whether to save params or not")
    parser.add_argument('--pretrain', help="Please give whether task is pretraing or not")
    parser.add_argument('--input_model_file', help="Please give input file name for pretraining")
    parser.add_argument('--output_model_file', help="Please give output file name for pretraining")

    #   additional arguments for mcdropout training
    parser.add_argument('--mcdropout', default=False, help="Please give a bool whether to use MCDropout or not")
    parser.add_argument('--mc_eval_num_samples', default=30, help="Please give an int for the number of samples used in inference")

    #   additional arguments for SWA/SWAG training

    parser.add_argument('--swa', default=False, help="Please give a bool whether to use SWA or not")
    parser.add_argument('--swag', default=False, help="Please give a bool whether to use SWAG or not")
    parser.add_argument('--swa_start', default=150, help="Please give an int value when to start swa moving averaging")
    parser.add_argument('--swa_lr_alpha1', default=0.01, help="Please give an int value when to start swa moving averaging")
    parser.add_argument('--swa_lr_alpha2', default=0.001, help="Please give an int value when to start swa moving averaging")
    parser.add_argument('--swa_c_epochs', default=4, help="Please give an int value when to start swa moving averaging")
    parser.add_argument('--swag_eval_scale', default=1., help="Please give an float value defining scale for swag inference")
    parser.add_argument('--swag_eval_num_samples', default=30, help="Please give an int value defining number of samples for swag inference")

    #   additional arguments for SGLD training

    parser.add_argument('--sgld', default=False, help="Please give a bool whether to use SWA or not")
    parser.add_argument('--psgld', default=False, help="Please give a bool whether to use SWA or not")
    parser.add_argument('--sgld_noise_std', default=0.001, help="Please give an int value when to start swa moving averaging")
    parser.add_argument('--sgld_save_every', default=2, help="Please give an int value when to start swa moving averaging")
    parser.add_argument('--sgld_save_start', default=100, help="Please give an int value when to start swa moving averaging")
    parser.add_argument('--sgld_max_samples', default=100, help="Please give an int value when to start swa moving averaging")

    #   additional arguments for BBP training

    parser.add_argument('--bbp', default=False, help="Please give a bool whether to use SWA or not")
    parser.add_argument('--bbp_complexity', default=0.1, help="Please give a float value defining complexity weight applied on KL loss")
    parser.add_argument('--bbp_sample_nbr', default=5, help="Please give an int value defining how much instances to be sampled for training")
    parser.add_argument('--bbp_eval_Nsample', default=100, help="Please give an int value defining how much instances to be sampled for evaluation")


    args = parser.parse_args()
    return args

def get_configs(args):
    
    with open(args.config) as f:
        config = json.load(f)
        
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True

    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])

    #   fix thread for reproducibility when using CPU
    print(config['gpu']['use'])
    '''
    if device != 'cuda':
       torch.set_num_threads(1)
    '''    

    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
        config['model'] = args.model
    else:
        MODEL_NAME = config['model']

    if args.dataset is not None:
        config['dataset'] = args.dataset
    else:
        args.dataset = config['dataset']

    # define num of train/val/test samples
    if args.num_train is not None:
        args.num_train = int(args.num_train)
        config['num_train'] = args.num_train
    else:
        args.num_train = config['num_train']
    if args.num_val is not None:
        args.num_val = int(args.num_val)
        config['num_val'] = args.num_val
    else:
        args.num_val = config['num_val']
    if args.num_test is not None:
        args.num_test = int(args.num_test)
        config['num_test'] = args.num_test
    else:
        args.num_test = config['num_test']

    if args.data_seed != None:
        config['data_seed'] = int(args.data_seed)
    else:
        args.data_seed = config['data_seed']

    # Atom Meta Info Usage
    if args.atom_meta is not None:
        if args.atom_meta == 'False':
            config['atom_meta'] = False
        else:
            config['atom_meta'] = True
    else:
        args.atom_meta = config['atom_meta']

    # Scaffold split Usage
    if args.scaffold_split is not None:
        if args.scaffold_split == 'True':
            config['scaffold_split'] = True
        else:
            config['scaffold_split'] = False

    else:
        config['scaffold_split'] = True

    #   save parameters config
    if 'save_params' not in [key for key in config.keys()]:
        config['save_params'] = False
    if args.save_params:
        if args.save_params == "True":
            config['save_params'] = True

    #   Parameters

    params = config['params']
    params['data_seed'] = config['data_seed']
    params['atom_meta'] = config['atom_meta']
    params['num_train'] = config['num_train']
    params['dataset'] = config['dataset']
    params['scaffold_split'] = config['scaffold_split']

    # Load data        
    if args.out_dir is not None:
        config['out_dir'] = args.out_dir
    else:
        args.out_dir = config['out_dir']

    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.scheduler is not None:
        params['scheduler'] = str(args.scheduler)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)

    if args.step_size is not None:
        params['step_size'] = int(args.step_size)
    else:
        params['step_size'] = 80
    if args.step_gamma is not None:
        params['step_gamma'] = float(args.step_gamma)
    else:
        params['step_gamma'] = 0.1

    #   bool for whether task is pretraining or not
    if args.pretrain is not None:
        params['pretrain'] = True
    else:
        params['pretrain'] = False

    params['optimizer'] = str(args.optimizer)
    params['grad_clip'] = float(args.grad_clip)
    
    #   MCDropout configs
    params['mcdropout'] = False
    if args.mcdropout:
        if args.mcdropout == "True":
            params['mcdropout'] = True
    if params['mcdropout'] == True:
       params['mc_eval_num_samples'] = int(args.mc_eval_num_samples)

    #   SWA configs
    params['swa'] = False
    if args.swa:
        if args.swa == "True":
            params['swa'] = True

    params['swag'] = False
    if args.swag:
        if args.swag == "True":
            params['swag'] = True

    if (params['swa'] == True) or (params['swag'] == True) :
        params['swa_start'] = int(args.swa_start)
        params['swa_lr_alpha1'] = float(args.swa_lr_alpha1)
        params['swa_lr_alpha2'] = float(args.swa_lr_alpha2)
        params['swa_c_epochs'] = int(args.swa_c_epochs)
        #   some default settings for SWA use
        params['init_lr'] = 0.1
        params['optimizer'] = "SGD"
        params['grad_clip'] = 1.0
    if params['swag'] == True:
        params['swag_eval_scale'] = float(args.swag_eval_scale)
        params['swag_eval_num_samples'] = int(args.swag_eval_num_samples)

    #   SGLD config
    params['sgld'] = False
    if args.sgld:
        if args.sgld == "True":
            params['sgld'] = True
    params['psgld'] = False
    if args.psgld:
        if args.psgld == "True":
            params['psgld'] = True
    if (params['sgld'] == True) or (params['psgld'] == True):
        params['sgld_noise_std'] = float(args.sgld_noise_std)
        params['sgld_save_every'] = int(args.sgld_save_every)
        params['sgld_save_start'] = int(args.sgld_save_start)
        params['sgld_max_samples'] = int(args.sgld_max_samples)

    #   BBP config
    params['bbp'] = False
    if args.bbp:
        if args.bbp == "True":
            params['bbp'] = True
    if params['bbp'] == True:
        params['bbp_complexity'] = float(args.bbp_complexity)
        params['bbp_sample_nbr'] = int(args.bbp_sample_nbr)
        params['bbp_eval_Nsample'] = int(args.bbp_eval_Nsample)
            
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    net_params['task'] = params['task']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)   
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated=='True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.graph_norm is not None:
        net_params['graph_norm'] = True if args.graph_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm=='True' else False
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop=='True' else False
        
    #   Additional arguments for GCN
    if args.agg is not None:
        if args.agg == 'mean':
            net_params['agg'] = 'mean'  
        else: 
            net_params['agg'] = 'sum'

    #   Additional arguments for GAT
    if args.att_reduce_fn is not None:
        if args.att_reduce_fn == 'tanh':
            net_params['att_reduce_fn'] = 'tanh'  
        else: 
            net_params['att_reduce_fn'] = 'softmax'
    #   Additional arguments for GIN
    if args.neighbor_aggr_GIN is not None:
        net_params['neighbor_aggr_GIN'] = str(args.neighbor_aggr_GIN)

    #   Additional arguments for GraphSage
    if args.concat_norm is not None:
        net_params['concat_norm'] = True if args.concat_norm=='True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = str(args.sage_aggregator)

    #   Additional arguments for GatedGCN
    if args.gated_gcn_agg is not None:
        net_params['gated_gcn_agg'] = str(args.gated_gcn_agg)
        
    if args.residual is not None:
        if args.residual=='gated':
            net_params['residual'] = 'gated'
        elif args.residual == 'True':
            net_params['residual'] = True
        else:
            net_params['residual'] = False

    return args, config, params, net_params

def add_dir_name(dirs, MODEL_NAME, config, params, net_params):
    
    out_dir = config['out_dir']

    root_ckpt_dir, write_file_name, root_output_dir = dirs
    """
        Write the results in out_dir/results folder
    """
    new_dirs = []
    for file_name in dirs: 
        file_name += ('_'+str(net_params['residual']))
        file_name += ('_'+str(net_params['readout']))

        if MODEL_NAME == 'GCN':
            file_name += ('_'+str(net_params['agg']))
             
        elif MODEL_NAME == 'GAT':
            file_name += ('_'+str(net_params['att_reduce_fn']))
            
        elif MODEL_NAME == 'GraphSage':
            file_name += ('_'+str(net_params['sage_aggregator']))

        elif MODEL_NAME == 'GatedGCN':
            file_name += ('_'+str(net_params['edge_feat']))

        if net_params['device'] == torch.device(type='cpu'):
            file_name = file_name.replace('GPU', 'CPU')

        if net_params['L'] != 4:
            file_name += ('_'+str(net_params['L']))
        
        if params['weight_decay'] != 0.:
            file_name += ('_'+str(params['weight_decay']))

        #   Bayesian Method stated
        if params['bbp'] == True:
            file_name += '_bbp'
        elif params['sgld'] == True:
            file_name += '_sgld'
        elif params['psgld'] == True:
            file_name += '_psgld'
        elif params['swa'] == True:
            file_name += '_swa'
        elif params['swag'] == True:
            file_name += '_swag'
        elif params['mcdropout'] == True:
            file_name += '_mcdropout'
        
        new_dirs.append(file_name)

    dirs = new_dirs[0], new_dirs[1], new_dirs[2]

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'checkpoints'):
        os.makedirs(out_dir + 'checkpoints')
        
    if not os.path.exists(out_dir + 'outputs'):
        os.makedirs(out_dir + 'outputs')

    return dirs
