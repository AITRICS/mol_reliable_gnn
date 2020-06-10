"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math

from train.metrics import binary_class_perfs 

#   import adjust lr for SWA cyclic lr
import swa_utils

def train_epoch_classification(model, optimizer, device, data_loader, epoch, params, cyclic_lr_schedule=None):
    model.train()
    epoch_loss = 0
    nb_data = 0
    gpu_mem = 0
    num_iters = len(data_loader.dataset)

    total_scores = []
    total_targets = []

    for iter, (batch_graphs, batch_targets, batch_snorm_n, batch_snorm_e, batch_smiles) in enumerate(data_loader):
        #   SWA cyclic lr schedule
        if cyclic_lr_schedule is not None:
            lr = cyclic_lr_schedule(iter / num_iters)
            swa_utils.adjust_learning_rate(optimizer, lr)


        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_snorm_e = batch_snorm_e.to(device)
        batch_targets = batch_targets.to(device)
        batch_snorm_n = batch_snorm_n.to(device)         # num x 1
        optimizer.zero_grad()
        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
            
        #   dtype for mse loss
        batch_targets = batch_targets.float()
        loss = model.loss(batch_scores, batch_targets)

        loss.backward()

        #   SGD with high lr show gradient explosion --> temporally using gradient clipping
        if params['grad_clip'] != 0.:
            clipping_value = params['grad_clip']
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)

        optimizer.step()
        epoch_loss += loss.detach().item()

        nb_data += batch_targets.size(0)

        total_scores.append(batch_scores)
        total_targets.append(batch_targets)


    epoch_loss /= (iter + 1)

    total_scores = torch.cat(total_scores, dim = 0)
    total_targets = torch.cat(total_targets, dim = 0)

    epoch_train_perf = binary_class_perfs(total_scores.detach(), total_targets.detach())
    
    return epoch_loss, epoch_train_perf, optimizer, total_scores, total_targets

def evaluate_network_classification(model, device, data_loader, epoch, params):
    

    if params['mcdropout'] == True:
        model.train()
    else:
        model.eval()

    epoch_test_loss = 0
    nb_data = 0

    total_scores = []
    total_targets = []
    total_smiles = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets, batch_snorm_n, batch_snorm_e, batch_smiles) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_targets = batch_targets.to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
            batch_targets = batch_targets.float()
            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()
            nb_data += batch_targets.size(0)
            total_scores.append(batch_scores)
            total_targets.append(batch_targets)
            total_smiles.extend(batch_smiles)

        epoch_test_loss /= (iter + 1)
        total_scores = torch.cat(total_scores, dim=0)
        total_targets = torch.cat(total_targets, dim=0)
        epoch_test_perfs = binary_class_perfs(total_scores, total_targets)

    return epoch_test_loss, epoch_test_perfs, total_scores, total_targets, total_smiles

