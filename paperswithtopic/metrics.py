from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn


def get_loss(pred, target, name):

    if name == 'mlml':
        return nn.MultiLabelMarginLoss(reduction='sum')(pred, target)

    elif name == 'bce':

        loss_fn = nn.BCELoss()
        loss = torch.tensor(0, device=pred.device).float()
        pred, target = pred.float(), target.float()
        num_col = pred.size()[1]
        for i in range(num_col):

            loss += loss_fn(pred[:, i], target[:, i])

        return loss / num_col


def get_auc(target, pred):

    return roc_auc_score(target, pred, multi_class='ovr')