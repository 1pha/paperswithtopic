from .config import load_config
import torch
import torch.nn as nn
from torchsummary import summary

from transformers.models.bert.modeling_bert import (
    BertModel, BertConfig, BertForSequenceClassification,
    AlbertModel, AlbertConfig, AlbertForSequenceClassification,
    ElectraModel, ElectraConfig, ElectraForSequenceClassification,
    XLMModel, XLMConfig, XLMForSequenceClassification,
)

def load_model(cfg):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Use {device} as a device.')
    print(f'Load {cfg.model_name.capitalize()} as model.')
    return {
        'rnn': RNN,
        'lstm': LSTM,
        'gru': GRU,
        'lstmattn': LSTMATTN,
        'bert': BERT,
        'albert': ALBERT,
        'electra': ELECTRA,
        'xlm': XLM,
        'pretrainedbert': PretrainedBERT,
        'bertclassification': BERTClassification,
        'albertclassification': ALBERTClassification,
        'electraclassification': ELECTRAClassification,
        'xlmclassification': XLMClassification,
    }[cfg.model_name.lower()](cfg).to(device), device


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
        self.n_layers   = cfg.n_layers

        if not self.cfg.pre_embed:
            self.embed_layer = nn.Embedding(self.vocab_size, self.hidden_dim)

        self.fc = nn.Linear(self.hidden_dim, self.num_class)
        self.relu = nn.ReLU()
        self.sfx = nn.Softmax(dim=1)


    def forward(self, x, mask):

        '''
        x:
            - (batch_size, seq_len) consists of indices
            or
            - (batch_size, seq_len, hidden_dim) each seq to a single vector

        mask:
            - (batch_size, seq_len)
            may not be used for some models
        '''

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

        self.config = BertConfig( 
            vocab_size=self.cfg.vocab_size,
            hidden_size=self.cfg.hidden_dim,
            num_hidden_layers=self.cfg.n_layers,
            num_attention_heads=self.cfg.n_heads,
            max_position_embeddings=self.cfg.MAX_LEN,
            num_labels=self.num_class,       
        )
        self.encoder = BertModel(self.config)


    def forward(self, x, mask):

        if not self.cfg.pre_embed: # given (batch_size, seq_len)

            # x = self.embed_layer(x)
            x = self.encoder(input_ids=x, attention_mask=mask)

        else: # given (batch_size, seq_len, hidden_dim)

            x = self.encoder(inputs_embeds=x, attention_mask=mask)

        '''
        TODO:: choose btw LAST_HIDDEN_STATE vs. POOLER_OUTPUT
            - LAST_HIDDEN_STATE (batch_size, seq_len, hidden_dim)
            - POOLER_OUTPUT (batch_size, hidden_dim) - use only the 1st sequence
        '''

        if self.cfg.which_output == 'pooler_output':
            
            # (batch_size, hidden_size)
            x = x['pooler_output']

        elif self.cfg.which_output == 'last_hidden_state':

            # (batch_size, hidden_size)
            x = x['last_hidden_state'][:, -1, :]

        x = self.fc(x)
        x = self.relu(x)
        x = self.sfx(x)

        return x


class PretrainedBERT(BERT):

    '''
    DEPRECATED: LACK OF VRAMS... SO SAD 
    '''

    def __init__(self, cfg):
        super().__init__(cfg)

        self.encoder = BertModel.from_pretrained('bert-base-uncased')


class ALBERT(BERT):

    '''
    ALBERT OUTPUTS
        - last_hidden_state (batch_size, seq_len, hidden_dim)
        - pooler_output (batch_size, hidden_size)
    '''

    def __init__(self, cfg):
        super().__init__(cfg)

        self.config = AlbertConfig( 
            vocab_size=self.cfg.vocab_size,
            hidden_size=self.cfg.hidden_dim,
            num_hidden_layers=self.cfg.n_layers,
            num_attention_heads=self.cfg.n_heads,
            max_position_embeddings=self.cfg.MAX_LEN,
            num_labels=self.num_class,       
        )
        self.encoder = AlbertModel(self.config)


class ELECTRA(BERT):

    '''
    ELECTRA OUTPUTS
        - last_hidden_state (batch_size, seq_len, hidden_dim)
        - NO POOLER OUTPUT
    '''

    def __init__(self, cfg):
        super().__init__(cfg)

        self.config = ElectraConfig( 
            vocab_size=self.cfg.vocab_size,
            hidden_size=self.cfg.hidden_dim,
            num_hidden_layers=self.cfg.n_layers,
            num_attention_heads=self.cfg.n_heads,
            max_position_embeddings=self.cfg.MAX_LEN,
            num_labels=self.num_class,       
        )
        self.encoder = ElectraModel(self.config)


class XLM(BERT):

    '''
    XLM OUTPUTS
        - last_hidden_state (batch_size, seq_len, hidden_dim)
        - NO POOLER OUTPUT
    '''

    def __init__(self, cfg):
        super().__init__(cfg)

        self.config = XLMConfig( 
            vocab_size=self.cfg.vocab_size,
            hidden_size=self.cfg.hidden_dim,
            num_hidden_layers=self.cfg.n_layers,
            num_attention_heads=self.cfg.n_heads,
            max_position_embeddings=self.cfg.MAX_LEN,
            num_labels=self.num_class,       
        )
        self.encoder = XLMModel(self.config)
        

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
        self.encoder = BertForSequenceClassification(self.config)


    def forward(self, x, mask):

        '''
        OVERRIDDEN
            since BertClassifier outputs (batch_size, num_classes)
            which consists of probability for each class
        '''

        if not self.cfg.pre_embed: # given (batch_size, seq_len)

            if not self.cfg.use_bert_embed: # USE TORCH EMBEDDING LAYER
                x = self.embed_layer(x)
                x = self.encoder(inputs_embeds=x, attention_mask=mask)

            else:
                x = self.encoder(input_ids=x, attention_mask=mask)

        else: # given (batch_size, seq_len, hidden_dim)
            
            x = self.encoder(inputs_embeds=x, attention_mask=mask)
        
        x = x['logits']
        x = self.sfx(x)

        return x


class ALBERTClassification(BERTClassification):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.config = AlbertConfig( 
            vocab_size=self.cfg.vocab_size,
            hidden_size=self.cfg.hidden_dim,
            num_hidden_layers=self.cfg.n_layers,
            num_attention_heads=self.cfg.n_heads,
            max_position_embeddings=self.cfg.MAX_LEN,
            num_labels=self.num_class,       
        )
        self.encoder = AlbertForSequenceClassification(self.config)


class ELECTRAClassification(BERTClassification):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.config = ElectraConfig( 
            vocab_size=self.cfg.vocab_size,
            hidden_size=self.cfg.hidden_dim,
            num_hidden_layers=self.cfg.n_layers,
            num_attention_heads=self.cfg.n_heads,
            max_position_embeddings=self.cfg.MAX_LEN,
            num_labels=self.num_class,       
        )
        self.encoder = ElectraForSequenceClassification(self.config)


class XLMClassification(BERTClassification):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.config = XLMConfig( 
            vocab_size=self.cfg.vocab_size,
            hidden_size=self.cfg.hidden_dim,
            num_hidden_layers=self.cfg.n_layers,
            num_attention_heads=self.cfg.n_heads,
            max_position_embeddings=self.cfg.MAX_LEN,
            num_labels=self.num_class,       
        )
        self.encoder = XLMForSequenceClassification(self.config)
    

class RNN(SequenceModel):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.seq_model = nn.RNN(
            input_size =self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers =self.n_layers,
            batch_first=True
        )

    def forward(self, x, mask):

        if not self.cfg.pre_embed:
            x = self.embed_layer(x)

        x , _= self.seq_model(x)

        x = self.fc(x)
        x = self.relu(x)
        x = self.sfx(x)

        return x[:, -1, :]
        

class LSTM(RNN):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.seq_model = nn.LSTM(
            input_size =self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers =self.n_layers,
            batch_first=True
        )


class GRU(RNN):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.seq_model = nn.GRU(
            input_size =self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers =self.n_layers,
            batch_first=True
        )


class LSTMATTN(RNN):

    def __init__(self, cfg):
        super().__init__(cfg)


if __name__=="__main__":

    cfg = load_config()

    model = BERTClassification(cfg)
    sample = torch.ones((2, cfg.MAX_LEN), dtype=torch.long)
    print(model(sample).shape)