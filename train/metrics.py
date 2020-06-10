import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score
import numpy as np

#   remove numpy overflow error from np_sigmoid
np.seterr(over='ignore')

def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    # MAE = torch.mean(torch.abs((scores - targets)))
    return MAE

def MSE(scores, targets):
    MSE = F.mse_loss(scores, targets)
    # MSE = torch.mean(torch.pow((scores - targets), 2))
    return MSE


def calibration_guo(label, pred, bins=10):
    
    width = 1.0 / bins
    bin_centers = np.linspace(0, 1.0-width, bins) + width/2

    conf_bin = []
    acc_bin = []
    conf_bin2 = []
    acc_bin2 = []
    counts = []
    for i, threshold in enumerate(bin_centers):
        bin_idx = np.logical_and(threshold - width/2 < pred,
                                 pred <= threshold + width)
        conf_mean = pred[bin_idx].mean()
        conf_sum = pred[bin_idx].sum()
        if (conf_mean != conf_mean) == False:
            conf_bin.append(conf_mean)
            conf_bin2.append(conf_sum)
            counts.append(pred[bin_idx].shape[0])
        
        acc_mean = label[bin_idx].mean()        
        acc_sum = label[bin_idx].sum()
        if (acc_mean != acc_mean) == False:
            acc_bin.append(acc_mean)
            acc_bin2.append(acc_sum)
    
    conf_bin = np.asarray(conf_bin)   
    acc_bin = np.asarray(acc_bin)   
    conf_bin2 = np.asarray(conf_bin2)   
    acc_bin2 = np.asarray(acc_bin2)   
    counts = np.asarray(counts)   

    ece = np.abs(conf_bin - acc_bin)
    ece = np.multiply(ece, counts)
    ece = ece.sum()
    ece /= np.sum(counts)
    # ece *= 100.0

    # oce = np.multiply(np.multiply(np.max(conf_bin-acc_bin,0), conf_bin), counts).sum()
    # oce /= np.sum(counts)
    # oce *= 100.0
    oce = None

    return conf_bin, acc_bin, ece, oce

def binary_class_perfs(scores, targets):
    """
    Input: predicted logits for positive class, before getting sigmoid fn.
    Output: a dict containing performances in float values.
    """

    if type(scores) == torch.Tensor:
        scores = scores.cpu().detach().numpy()
    if type(targets) == torch.Tensor:
        targets = targets.cpu().detach().numpy()

    fn = lambda x: 1 if x>0 else 0
    preds = np.array([fn(x) for x in scores])

    def np_sigmoid(x):
        return 1./(1. + np.exp(-x))

    perfs = {}
    perfs['accuracy'] = accuracy_score(targets, preds)
    perfs['precision'] = precision_score(targets, preds)
    perfs['recall'] = recall_score(targets, preds)
    perfs['f1'] = f1_score(targets, preds)
    try:
        perfs['auroc'] = roc_auc_score(targets, np_sigmoid(scores))
        perfs['auprc'] = average_precision_score(targets, np_sigmoid(scores))
    except:
        perfs['auroc'] = 0.
        perfs['auprc'] = 0.

    _, _, perfs['ece'], _ = calibration_guo(targets, np_sigmoid(scores), bins=10)

    return perfs


