from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader


class PaperDataset(Dataset):

    def __init__(self, cfg, X=None, y=None):

        # SETUP
        self.cfg = cfg
        self.pad = cfg.PAD
        self.seq_len = cfg.MAX_LEN
        SEED = cfg.seed
        if X is None or y is None:
            print(f'Please feed data.')
            
        # DATA
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size, random_state=SEED)
        if not cfg.test:
            self.data, self.label = X_train, y_train
             
        else:
            self.data, self.label = X_test, y_test


    def __getitem__(self, idx):

        paper, label = torch.tensor(self.data[idx]), torch.tensor(self.label[idx])

        if not self.pad:
            return paper, label

        mask = torch.ones(self.seq_len)
        if sum(paper != 0) < self.seq_len:
            mask[sum(paper != 0):] = 0

        return paper, label, mask

    