import os
import yaml
import torch

from .args import parse_args
from .config import load_config
from .preprocess import Preprocess
from .model import load_model
from .misc import logging_time


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


if __name__=="__main__":

    args = parse_args()
    cfg = load_config()
    cfg.update(args)

    paper = cfg.query
    print(inference(cfg, paper))