from functools import partial
from sklearn.metrics import roc_auc_score
import torch.nn as nn


def get_loss(pred, target, name):
    
    return {
        'mlml': nn.MultiLabelMarginLoss(reduction='none')
    }[name](pred, target)


def get_auc(target, pred):

    return roc_auc_score(target, pred, multi_class='ovr')