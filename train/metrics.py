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
    # if targets.dtype == "float32":
    #     targets = targets.astype("int32")


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



'''
def accuracy_TU(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    return acc


def accuracy_MNIST_CIFAR(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    return acc


def accuracy_SBM(scores, targets):
    S = targets.cpu().numpy()
    C = np.argmax( torch.nn.Softmax(dim=0)(scores).cpu().detach().numpy() , axis=1 )
    CM = confusion_matrix(S,C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets==r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r,r]/ float(cluster.shape[0])
            if CM[r,r]>0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = 100.* np.sum(pr_classes)/ float(nb_non_empty_classes)
    return acc


def binary_f1_score(scores, targets):
    """Computes the F1 score using scikit-learn for binary class labels. 
    
    Returns the F1 score for the positive class, i.e. labelled '1'.
    """
    y_true = targets.cpu().numpy()
    y_pred = scores.argmax(dim=1).cpu().numpy()
    return f1_score(y_true, y_pred, average='binary')

  
def accuracy_VOC(scores, targets):
    scores = scores.detach().argmax(dim=1).cpu()
    targets = targets.cpu().detach().numpy()
    acc = f1_score(scores, targets, average='weighted')
    return acc

def class_perfs(scores, targets):
    scores = scores.cpu().numpy()
    preds = np.argmax(scores, axis = -1)
    targets = targets.cpu().detach().numpy()

    perfs = {}
    perfs['accuracy'] = accuracy_score(targets, preds)
    perfs['precision'] = precision_score(targets, preds)
    perfs['recall'] = recall_score(targets, preds)
    perfs['f1'] = f1_score(targets, preds)
    perfs['auroc'] = roc_auc_score(targets, scores[:, 1]) 
    perfs['auprc'] = average_precision_score(targets, scores[:, 1]) 
    
    return perfs

def accuracy_Mol(scores, targets):
    scores = scores.detach().cpu().numpy()
    fn = lambda x: 1 if x>0 else 0
    preds = np.array([fn(x) for x in scores])
    targets = targets.detach().cpu().numpy()
    #acc = np.sum((preds==targets))
    acc = accuracy_score(targets, preds)
    return acc
'''
