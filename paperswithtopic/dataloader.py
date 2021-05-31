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
            
        # DATA
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size, random_state=SEED)
        if not self.test:
            self.data, self.label = X_train, y_train
             
        else:
            self.data, self.label = X_test, y_test


    def __getitem__(self, idx):

        paper, label = torch.tensor(self.data[idx], type=torch.long), torch.tensor(self.label[idx], type=torch.long)

        if not self.pad:
            return paper, label

        mask = torch.ones(self.seq_len)
        if sum(paper != 0) < self.seq_len:
            mask[sum(paper != 0):] = 0

        return paper, label, mask

    def __len__(self):
        return len(self.data)

    
def get_dataloader(cfg, X, y, test, **kwargs):

    dataset = PaperDataset(cfg, X, y, test)
    dataloader = DataLoader(dataset, cfg.batch_size, pin_memory=True, **kwargs)

    return dataloader