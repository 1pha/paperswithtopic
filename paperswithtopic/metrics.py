from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn


def get_loss(pred, target, name):

    if name == 'mlml':
        return nn.MultiLabelMarginLoss(reduction='sum')(pred, target)

    if name == 'bce':

        loss_fn = nn.BCELoss()
        loss = torch.tensor(0, device=pred.device)
        for i in range(pred.size()[1]):

            loss += loss_fn(pred[:, i].float(), target[:, i].float())

        return loss


def get_auc(target, pred):

    return roc_auc_score(target, pred, multi_class='ovr')