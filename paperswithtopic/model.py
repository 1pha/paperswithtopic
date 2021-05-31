import torch
import torch.nn as nn

class NaiveBayes:

    def __init__(self, cfg):

        pass


class BERT(nn.Module):

    def __init__(self, cfg):

        pass

    def forward(x):

        pass
    
class RNN(nn.Module):

    def __init__(self, cfg):
        super().__init__(cfg)

        pass

    def forward(self, x):

        self.lstm = nn.LSTM(
            input_size =self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers =self.n_layers,
            batch_first=True
        )
        

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

class LSTM(RNN):

    def __init__(self, cfg):

        pass