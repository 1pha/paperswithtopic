import os
import numpy as np
from transformers.utils import logging
import wandb

import torch
from .preprocess import Preprocess
from .dataloader import get_dataloader
from .optimizer import get_optimizer, get_scheduler
from .model import load_model
from .metrics import get_loss, get_auc
from .misc import logging_time


def setup(cfg):

    preprocess = Preprocess(cfg=cfg)
    if cfg.use_saved:
        X = preprocess.load_processed()
        y = preprocess.label    
        word_mapper = preprocess.load_word_mapper()
        
    else:
        X, y = preprocess.pp_pipeline()
        word_mapper = preprocess.word_mapper
        
    cfg.vocab_size = len(word_mapper)
    train_dataloader = get_dataloader(cfg, X, y, test=False)
    valid_dataloader = get_dataloader(cfg, X, y, test=True)

    return train_dataloader, valid_dataloader


def run(cfg):

    train_dataloader, valid_dataloader = setup(cfg)
    print(f'NUM TRAIN {len(train_dataloader.dataset)} | NUM VALID {len(valid_dataloader.dataset)}')

    model, cfg.device = load_model(cfg)
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)

    best_auc = 0
    early_stopping_counter = 0
    for epoch in range(cfg.start_epoch, cfg.n_epochs):

        print(f'Epoch {epoch + 1} / {cfg.n_epochs}, BEST AUC {best_auc:.3f}')
        trn_auc, trn_loss, trn_pred = train(train_dataloader, model, optimizer, cfg)
        val_auc, val_loss, val_pred = valid(valid_dataloader, model, cfg)
        scheduler.step(best_auc)

        wandb.log({
            'train_auc': trn_auc,
            'valid_auc': val_auc,

            'train_loss': trn_loss,
            'valid_loss': val_loss,
        })

        print(f'TRAIN:: AUC {trn_auc:.3f} | LOSS {trn_loss:.3f}')
        print(f'VALID:: AUC {val_auc:.3f} | LOSS {val_loss:.3f}')

        if val_auc > best_auc:
            best_auc = val_auc
            early_stopping_counter = 0

        else:
            early_stopping_counter += 1
            if early_stopping_counter >= cfg.early_patience:
                print(f'EarlyStopping counter: {early_stopping_counter} out of {cfg.early_patience}')
                break

    cfg.best_auc = best_auc
    wandb.config.update(cfg)
    wandb.finish()

    return model


@logging_time
def train(dataloader, model, optimizer, cfg, total_targets=None):

    if total_targets is None:
        total_targets = dataloader.dataset.label

    model.train()
    total_preds = []
    losses = []
    for step, batch in enumerate(dataloader):

        paper, label, mask = map(lambda x: x.to(cfg.device), batch)
        preds = model(paper, mask)

        optimizer.zero_grad()
        loss = torch.sum(get_loss(preds, label, cfg.loss))
        loss.backward()
        losses.append(loss)
        optimizer.step()

        preds = preds.to('cpu').detach().numpy()
        total_preds.append(preds)        
      
    total_preds = np.concatenate(total_preds)
    total_targets = dataloader.dataset.label

    auc = get_auc(total_targets, total_preds)
    loss_avg = sum(losses) / len(losses)

    return auc, loss_avg, (total_preds, total_targets)


@logging_time
def valid(dataloader, model, cfg, total_targets=None):

    if total_targets is None:
        total_targets = dataloader.dataset.label

    model.eval()
    total_preds = []
    losses = []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):

            paper, label, mask = map(lambda x: x.to(cfg.device), batch)
            preds = model(paper, mask)

            loss = torch.sum(get_loss(preds, label, cfg.loss))

            preds = preds.to('cpu').detach().numpy()
        
            total_preds.append(preds)
            losses.append(loss)
      
    total_preds = np.concatenate(total_preds)

    auc = get_auc(total_targets, total_preds)
    loss_avg = sum(losses) / len(losses)

    return auc, loss_avg, total_preds


def save_checkpoint(state, model_dir, model_filename):
    print('saving model ...')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    
    torch.save(state, os.path.join(model_dir, model_filename))