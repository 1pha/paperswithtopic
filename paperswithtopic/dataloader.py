import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader


class PaperDataset(Dataset):

    def __init__(self, cfg, X=None, y=None, test=False):

        # SETUP
        self.cfg = cfg
        self.test = test
        self.pad = cfg.PAD
        self.seq_len = cfg.MAX_LEN
        SEED = cfg.seed
        if X is None or y is None:
            print(f'Please feed data.')

        if cfg.partial:
            X, y = X[:int(len(X) * cfg.partial)], y[:int(len(X) * cfg.partial)]
            
        # DATA
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size, random_state=SEED)
        if not self.test:
            self.data, self.label = X_train, y_train
             
        else:
            self.data, self.label = X_test, y_test


    def __getitem__(self, idx):

        paper, label = torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.label[idx], dtype=torch.long)

        if not self.pad:
            return paper, label

        mask = self.build_mask(paper)

        return paper, label, mask

    def __len__(self):
        return len(self.data)


    def build_mask(self, paper):

        if self.cfg.pre_embed:
            _paper = torch.sum(paper, axis=1)

        mask = torch.ones(self.seq_len)
        if sum(_paper != 0) < self.seq_len:
            mask[sum(_paper != 0):] = 0
        return mask        

    
def get_dataloader(cfg, X, y, test, **kwargs):

    dataset = PaperDataset(cfg, X, y, test)
    dataloader = DataLoader(dataset, cfg.batch_size, pin_memory=True, **kwargs)

    return dataloader