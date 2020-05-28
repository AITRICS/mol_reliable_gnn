import os
import torch
import copy

"""
Ref: https://github.com/timgaripov/swa
"""

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha

def save_checkpoint(root_ckpt_dir, epoch, params, **kwargs):

    ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = ckpt_dir + '/seed_' + str(params['seed']) + '_dtseed_' + str(params['data_seed']) + '_epoch_' + str(epoch) + '.pt'
    torch.save(state, filepath)

def schedule(epoch, params):
    t = (epoch) / (params['swa_start'] if params['swa'] else params['epochs'])
    lr_ratio = params['swa_lr_alpha1'] / params['init_lr'] if params['swa'] else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return params['init_lr'] * factor

def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    """
    Ref: https://github.com/timgaripov/dnn-mode-connectivity
    """
    
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule

