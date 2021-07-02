import os
import yaml
import torch

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from .args import parse_args
from .config import load_config
from .preprocess import Preprocess
from .model import load_model
from .misc import logging_time
from .optimizer import get_optimizer
from .metrics import get_loss


@logging_time
def inference(cfg, papers: list, model_path=None):

    '''
    paper: raw list of paper titles
    # TODO: cfg.pre_embed currently does not work
    '''

    cfg.num_class = 16 - len(cfg.drop)
    model, cfg.device = load_model(cfg)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    preprocess = Preprocess(cfg=cfg)
    batch = preprocess.preprocess_infer(papers)
    papers, mask = map(lambda x: x.to(cfg.device), batch)
    
    model.eval()
    with torch.no_grad():

        if cfg.output_attentions:
            pred, attn = model(papers, mask)
            pred = pred.to('cpu').detach().numpy()
            return pred, attn

        else:
            pred = model(papers, mask)
            pred = pred.to('cpu').detach().numpy()
            return pred


@logging_time
def _inference(cfg, papers: list, model_path=None):

    '''
    This function is made to do afterworks for the project
    '''

    cfg.num_class = 16 - len(cfg.drop)
    model, cfg.device = load_model(cfg)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        print('Model loaded')

    preprocess = Preprocess(cfg=cfg)
    batch = preprocess.preprocess_infer(papers)
    papers, mask = map(lambda x: x.to(cfg.device), batch)
    
    model.eval()
    with torch.no_grad():

        if cfg.output_attentions:
            pred, attn = model(papers, mask)
            pred = pred.to('cpu').detach().numpy()
            attn = tuple(map(lambda x: x.cpu().detach().numpy(), attn))
            return pred, attn, (papers.cpu().detach().numpy(), mask.cpu().detach().numpy())

        else:
            pred = model(papers, mask)
            pred = pred.to('cpu').detach().numpy()
            return pred, (papers.cpu().detach().numpy(), mask.cpu().detach().numpy())

    

@logging_time
def inference_with_model(cfg, papers, model):

    preprocess = Preprocess(cfg=cfg)
    batch = preprocess.preprocess_infer(papers)
    papers, mask = map(lambda x: x.to(cfg.device), batch)

    with torch.no_grad():
        pred = model(papers, mask)
        pred = pred.to('cpu').detach().numpy()

    return pred

def revert2class(preds, cfg, topn=3):

    with open(os.path.join(cfg.DATA_DIR, 'column2idx.yml'), 'r') as f:
        column2idx = yaml.load(f)
    
    for d in cfg.drop:
        del column2idx[d]
    column2idx = {i: k for i, k in enumerate(column2idx.keys())}
    idx = preds.argsort()[:, -topn:][:, ::-1]

    predictions, probas = [], []
    for i, _idx in enumerate(idx):
        
        predictions.append(list(map(lambda i: column2idx[i], _idx)))
        probas.append(preds[i][_idx].tolist())

    return predictions, probas
    

def class2label(labels: list, cfg):

    with open(os.path.join(cfg.DATA_DIR, 'column2idx.yml'), 'r') as f:
        column2idx = yaml.load(f)

    idx = list(map(lambda x: column2idx[x], labels))
    return torch.tensor([[0 if i not in idx else 1 for i in range(len(column2idx))]])


def return_gradient(paper, labels, cfg, model_path):

    if isinstance(labels, str):
        labels = [labels]
    
    if isinstance(paper, str):
        paper = [paper]

    cfg.num_class = 16 - len(cfg.drop)
    model, cfg.device = load_model(cfg)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        print('Model loaded')

    # SETUP
    optimizer = get_optimizer(model, cfg)
    preprocess = Preprocess(cfg=cfg)
    batch = preprocess.preprocess_infer(paper)
    print(batch[0])
    papers, mask = map(lambda x: x.to(cfg.device), batch)

    # PREDICT
    with torch.no_grad():
        model.eval()
        pred, attn = model(papers, mask)

    result = revert2class(pred.detach().cpu().numpy(), cfg)
    print(result)

    # FORWARD
    model.train()
    pred, attn = model(papers, mask)
    optimizer.zero_grad()
    loss = get_loss(pred, labels, cfg.loss)

    # BACKWARD
    loss.backward()

    grad = model.gradient[0][0][:int(mask.sum())].detach().cpu().numpy()
    input_grad = np.sum(np.square(grad), axis=1) / np.sum(np.square(grad))

    xticks = preprocess.remove_unknown(paper)[0].split()
    return xticks, result, input_grad

def plot_gradient(xticks, result, input_grad):

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_title(f'Model Top3 Prediction: {", ".join(result[0][0])}', size='x-large')
    ax.set_ylabel('L2 Norm Gradient', size='x-large')
    ax.set_xticklabels(xticks, size='x-large', rotation=45)
    plt.bar(xticks, input_grad)
    plt.savefig('fig3a.png', transparent=False, facecolor='white', bbox_inches='tight')


if __name__=="__main__":

    args = parse_args()
    cfg = load_config()
    cfg.update(args)

    paper = cfg.query
    print(inference(cfg, paper))