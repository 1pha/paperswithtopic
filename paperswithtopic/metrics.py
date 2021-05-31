import torch.nn as nn

def get_loss(pred, target, name):
    
    return {
        'mlml': nn.MultiLabelMarginLoss(reduction='none')
    }[name](pred, target)