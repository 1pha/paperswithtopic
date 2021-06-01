import os
import numpy as np

import torch
from .preprocess import Preprocess
from .dataloader import get_dataloader
from .optimizer import get_optimizer, get_scheduler
from .model import load_model
from .metrics import get_loss, get_auc


def data_setup(cfg):

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

    train_dataloader, valid_dataloader = data_setup(cfg)
    print(f'NUM TRAIN {len(train_dataloader.dataset)} | NUM VALID {len(valid_dataloader.dataset)}')

    model = load_model(cfg)
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)

    best_auc = 0
    early_stopping_counter = 0
    for epoch in range(cfg.start_epoch, cfg.n_epochs):

        trn_auc, trn_loss = train(train_dataloader, model, optimizer, cfg)
        val_auc, val_loss = valid(valid_dataloader, model, cfg)

        print(f'TRAIN:: AUC {trn_auc} | LOSS {trn_loss}')
        print(f'VALID:: AUC {val_auc} | LOSS {val_loss}')

        # wandb.log({"epoch": epoch, "train_loss": train_loss, "train_auc": train_auc, "train_acc":train_acc,
        #     "valid_auc":auc, "valid_acc":acc})
        if val_auc > best_auc:
            best_auc = val_auc
            early_stopping_counter = 0

        else:
            early_stopping_counter += 1
            if early_stopping_counter >= cfg.patience:
                print(f'EarlyStopping counter: {early_stopping_counter} out of {cfg.patience}')
                break

        # scheduler
        if cfg.scheduler == 'plateau':
            scheduler.step(best_auc)
        else:
            scheduler.step()

    return model

    
def train(dataloader, model, optimizer, cfg):

    model.train()

    total_preds = []
    losses = []
    for step, batch in enumerate(dataloader):

        paper, label, mask = batch
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

    return auc, loss_avg


def valid(dataloader, model, cfg):

    model.eval()

    total_preds = []
    losses = []
    for step, batch in enumerate(dataloader):

        paper, label, mask = batch
        preds = model(paper, mask)

        loss = torch.sum(get_loss(preds, label, cfg.loss))

        preds = preds.to('cpu').detach().numpy()
    
        total_preds.append(preds)
        losses.append(loss)
      
    total_preds = np.concatenate(total_preds)
    total_targets = dataloader.dataset.label

    auc = get_auc(total_targets, total_preds)
    loss_avg = sum(losses) / len(losses)

    return auc, loss_avg


def save_checkpoint(state, model_dir, model_filename):
    print('saving model ...')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    
    torch.save(state, os.path.join(model_dir, model_filename))