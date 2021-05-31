from functools import partial
from sklearn.metrics import roc_auc_score
import torch.nn as nn

def get_loss(pred, target, name):
    
    return {
        'mlml': nn.MultiLabelMarginLoss(reduction='none')
    }[name](pred, target)

def get_metric(pred, target, name):

    return {
        'auc': partial(roc_auc_score, multi_class='ovr')
    }[name](target, pred)