"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math

from train.metrics import MAE, MSE

def pretrain_epoch(args, model, device, loader, optimizer):
    model.train()

    for step, (batch_graphs, batch_targets, batch_snorm_n, batch_snorm_e, batch_smiles) in enumerate(loader):
        batch_x = batch_graphs.ndata['feat'].to(device) # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_snorm_e = batch_snorm_e.to(device)
        batch_targets = batch_targets.to(device)
        batch_snorm_n = batch_snorm_n.to(device)

        pred = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
        y = batch_targets.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = model.loss(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def pre_evaluate_network(model, device, data_loader, epoch, params):
    model.eval()
    epoch_test_loss = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_targets = batch_targets.to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
            if params['task'] == 'classification':
                batch_targets = batch_targets.long().squeeze(-1)
            else:
                batch_targets = batch_targets.float()
            
            if len(batch_targets.shape) == 3:
                batch_targets = batch_targets.squeeze(1).float()

            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()

        epoch_test_loss /= (iter + 1)
        
    return epoch_test_loss

def train_epoch_regression(model, optimizer, device, data_loader, epoch, params):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    gpu_mem = 0

    total_scores = []
    total_targets = []

    for iter, (batch_graphs, batch_targets, batch_snorm_n, batch_snorm_e, batch_smiles) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_snorm_e = batch_snorm_e.to(device)
        batch_targets = batch_targets.to(device)
        batch_snorm_n = batch_snorm_n.to(device)         # num x 1
        optimizer.zero_grad()
        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)

        #   dtype for mse loss
        batch_targets = batch_targets.type(torch.float32)

        loss = model.loss(batch_scores, batch_targets)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()

        #   Model.loss is set to mse loss ATM.
        epoch_train_mae += MAE(batch_scores, batch_targets)

        total_scores.append(batch_scores)
        total_targets.append(batch_targets)

        nb_data += batch_targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)

    total_scores = torch.cat(total_scores, dim = 0)
    total_targets = torch.cat(total_targets, dim = 0)
    
    return epoch_loss, epoch_train_mae, optimizer, total_scores, total_targets

def evaluate_network_regression(model, device, data_loader, epoch, params):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
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
            epoch_test_mae += MAE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
            total_scores.append(batch_scores)
            total_targets.append(batch_targets)
            total_smiles.extend(batch_smiles)

        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)

        total_scores = torch.cat(total_scores, dim = 0)
        total_targets = torch.cat(total_targets, dim = 0)

        
    return epoch_test_loss, epoch_test_mae, total_scores, total_targets, total_smiles
