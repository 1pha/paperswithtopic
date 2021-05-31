from .config import load_config
import torch
import torch.nn as nn
from torchsummary import summary

from transformers.models.bert.modeling_bert import BertModel, BertConfig, BertForSequenceClassification

def load_model(cfg):

    return {
        'rnn': RNN,
        'lstm': LSTM,
        'gru': GRU,
        'lstmattn': LSTMATTN,
        'bert': BERT,
        'bertclassification': BERTClassification,
    }

class NaiveBayes:

    def __init__(self, cfg):

        pass


class SequenceModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.hidden_dim = cfg.hidden_dim
        self.num_class  = cfg.num_class

        if not self.cfg.pre_embed:
            self.embed_layer = nn.Embedding(self.vocab_size, self.hidden_dim)

        self.fc = nn.Linear(self.hidden_dim, self.num_class)
        self.relu = nn.ReLU()
        self.sfx = nn.Softmax(dim=1)


    def forward(self, x):

        if not self.cfg.pre_embed:
            x = self.embed_layer(x)

        x = self._forward(x)

        x = self.fc(x)
        x = self.relu(x)
        x = self.sfx(x)

        return x


    def _forward(self, x):

        '''
        SHOULD BE IMPLEMENTED FOR EACH MODEL
        '''

        return x

class BERT(SequenceModel):
    
    def __init__(self, cfg):
        super().__init__(cfg)


class BERTClassification(SequenceModel):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.config = BertConfig( 
            vocab_size=self.cfg.vocab_size,
            hidden_size=self.cfg.hidden_dim,
            num_hidden_layers=self.cfg.n_layers,
            num_attention_heads=self.cfg.n_heads,
            max_position_embeddings=self.cfg.MAX_LEN,
            num_labels=self.num_class,       
        )
        # self.encoder = BertModel(self.config)  
        self.encoder = BertForSequenceClassification(self.config)


    def forward(self, x):

        '''
        OVERRIDE
        '''

        if not self.cfg.pre_embed:
            x = self.embed_layer(x)
            x = self.encoder(inputs_embeds=x)
            x = x['logits']
            x = self.sfx(x)

        return x

    
class RNN(SequenceModel):

    def __init__(self, cfg):
        super().__init__(self, cfg)

        self.lstm = nn.LSTM(
            input_size =self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers =self.n_layers,
            batch_first=True
        )

    def forward(self, x):

        pass
    

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
        super().__init__(cfg)


class GRU(RNN):

    def __init__(self, cfg):
        super().__init__(cfg)


class LSTMATTN(RNN):

    def __init__(self, cfg):
        super().__init__(cfg)


if __name__=="__main__":

    cfg = load_config()

    model = BERTClassification(cfg)
    sample = torch.ones((2, cfg.MAX_LEN), dtype=torch.long)
    print(model(sample).shape)