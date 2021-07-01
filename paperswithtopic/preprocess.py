import os
import yaml
import torch
import numpy as np
import pandas as pd
try:
    from gensim.models import FastText, Word2Vec
except:
    pass

from .config import load_config
from .misc import logging_time


class Preprocess:

    def __init__(self, cfg=None):

        if cfg is None:
            cfg = load_config()

        self.cfg = cfg
        
        self._pad = '<pad>'
        self._unk = '<unk>'


    @logging_time
    def load_data(self, path=None):

        if path is None:
            path = os.path.join(self.cfg.DATA_DIR, self.cfg.file_name)

        df = pd.read_csv(path, index_col=0)
        df[df >= 1] = 1
        df = df.astype({c: 'int8' for c in df.columns})

        self.paper_df = df

        return df


    @property
    def label(self):
        try: 
            return self.paper_df.values
        
        except:
            self.load_data()
            return self.paper_df.values

    
    @logging_time
    def build_mask(self, paper: list): 

        '''
        paper: (batch_size, max_seq_len) list consists with index.
        > outputs (batch_size, max_seq_len) of masks
        '''

        mask = torch.ones(self.cfg.MAX_LEN)
        if paper.index(0) < self.cfg.MAX_LEN:
            mask[paper.index(0):] = 0

        return mask


    @logging_time
    def preprocess_infer(self, paper):

        if not hasattr(self, 'idx2word'):
            self.idx2word = self.load_idx2word()

        if isinstance(paper, str):
            paper = [paper]

        paper = self.remove_unknown(paper)
        paper = self.tokenize_papers(paper)
        mask = torch.stack(list(map(self.build_mask, paper)))

        if self.cfg.pre_embed:
            if not hasattr(self, 'gensim'):
                self.load_fasttext()
            paper = self.embed_gensim(paper)

        return torch.tensor(paper), mask


    @logging_time
    def pp_pipeline(self, path=None, return_y=True):

        '''
        1. LOAD RAW PAPERS
        4. REMOVE UNKNOWNS
        5. BUILD WORD MAPPER
        6. TOKENIZE (WORD2IDX)
        '''

        # 0. LOAD DATAFRAME
        df = self.load_data(path)

        # 1. LOAD RAW PAPERS
        X = self.retrieve_raw_papers(df) # -> X_raw
        X = self.remove_unknown(X) # -> X_filter
        self.build_idx2word(X)

        # +. EMBED PAPERS
        if self.cfg.pre_embed:
            self.gensim = self.train_embed(X=X, embed_dim=self.cfg.embed_dim, window=self.cfg.MAX_LEN)
            X = self.embed_gensim(X)

        else: 
            X = self.tokenize_papers(X, self.cfg.PAD)

        if not return_y:
            return X
        else:
            y = self.drop_columns(columns=self.cfg.drop)
            return X, y


    @logging_time
    def tokenized_pipeline(self):

        fname = f'X_tokenized.npy'
        X = self.load_tokenized(fname)
        y = self.drop_columns(columns=self.cfg.drop)

        idx2word = self.load_idx2word()
        return X, y, idx2word


    @logging_time
    def preembed_pipeline(self):

        fname = f'X_embed{self.cfg.embed_dim}.npy'
        X = self.load_embedded(fname)
        y = self.drop_columns(columns=self.cfg.drop)

        idx2word = self.load_idx2word()
        return X, y, idx2word


    @logging_time
    def retrieve_raw_papers(self, df):

        if df is None:
            self.df = df

        self.paper2idx = {p: i for p, i in enumerate(df.index)}
        self.X_raw = list(map(str.lower, self.paper2idx.values()))
        print(f'There are {len(self.X_raw)} papers.')
    
        return self.X_raw


    @logging_time
    def remove_unknown(self, X=None):

        if X is None:
            X = self.X_raw

        _remove = lambda sentence: ''.join([l for l in sentence if l.isalnum() or l == ' '])
        self.X_filter = list(map(_remove, X))
        return self.X_filter


    @logging_time
    def build_idx2word(self, X=None):

        if X is None:
            X = self.X_filter

        self.idx2word = {0: self._pad, 1: self._unk}
        def _encode(paper, idx):
            is_valid = lambda l: l.isalnum() or l == ' '
            
            for word in list(filter(lambda x: x, paper.lower().split(' '))):
                
                if word in self.idx2word.values():
                    pass
                
                elif all(filter(is_valid, word)):
                    self.idx2word[idx] = word
                    idx += 1
                    
            return idx

        idx = 2
        for paper in X:
            idx = _encode(paper, idx)

        self.save_idx2word()
        return self.idx2word


    @logging_time
    def tokenize_papers(self, X=None, pad=True, idx2word=None):

        '''
        This will map word > index
        If pad is True, it will make every sequence to maximum sequence,
        so shorter sequences would have 0-pads in the back (post-padding).
        '''

        if X is None:
            X = self.X_filter

        if idx2word is None:
            idx2word = self.idx2word

        self.word2idx = {v: k for k, v in self.idx2word.items()}
        def _tokenize(paper):
            
            words = list(filter(lambda x: x, paper.lower().split(' ')))
            SEQ_LEN = self.cfg.MAX_LEN if pad else len(words)
            tokens = [0 for _ in range(SEQ_LEN)]
            for idx in range(min(len(words), SEQ_LEN)):
                try:
                    tokens[idx] = self.word2idx[words[idx]]
                except:
                    tokens[idx] = self.word2idx['<unk>']
                
            return tokens

        self.X_tokenized = [_tokenize(paper) for paper in X]
        return self.X_tokenized


    @logging_time
    def train_embed(self, X=None, embed_dim=None, **kwargs):

        if X is None:
            X = self.X_filter

        if embed_dim is None:
            embed_dim = self.cfg.embed_dim

        X_fasttext = list(map(str.split, X))
        print(f'Use {self.cfg.pre_embed} as embedding')
        if self.cfg.pre_embed == 'fasttext':
            return FastText(sentences=X_fasttext, vector_size=embed_dim, **kwargs)
        
        elif self.cfg.pre_embed == 'word2vec':
            return Word2Vec(sentences=X_fasttext, vector_size=embed_dim, min_count=0, **kwargs)


    @logging_time
    def embed_gensim(self, X=None, model=None, **kwargs):

        if X is None:
            X = self.X_filter

        if model is None:
            model = self.gensim

        embed_dim = model.wv.vectors.shape[1]
        def embed(word): # EMBED A SINGLE WORD TO EMBEDDED VECTOR
            
            if word == '<pad>': # IF PAD, JUST RETURN 0 VECTOR
                return np.zeros((1, embed_dim))
            
            else: # USE MODEL TO GET EMBED VECTORS
                return model.wv.get_vector(word).reshape(1, -1)
        
        X_embed = [0 for _ in X]
        for i, paper in enumerate(X):

            paper = paper.split()
            if len(paper) >= self.cfg.MAX_LEN:
                paper = paper[:self.cfg.MAX_LEN]

            else:
                paper = paper + [0] * (self.cfg.MAX_LEN - len(paper))
                
            X_embed[i] = np.concatenate(list(map(embed, paper)))

        return np.array(X_embed)


    def drop_columns(self, columns=None):

        if columns is not None:
            
            with open('./data/column2idx.yml', 'r') as f:
                column2idx = yaml.load(f, Loader=yaml.FullLoader)

            unused_columns = [column2idx[c] for c in columns]
            used_columns = [i for i in range(16) if i not in unused_columns]  
            self.cfg.num_class = len(used_columns)

            return self.label[:, used_columns]

        else: 
            return self.label

    
    def save_data(self, X):

        np.save(os.path.join(self.cfg.DATA_DIR, 'X_tokenized.npy'), X)


    def save_idx2word(self, fname=None):

        if fname is None:
            fname = 'idx2word.yml'

        with open(os.path.join(self.cfg.DATA_DIR, fname), 'w') as y:
            yaml.dump(self.idx2word, y)


    def save_fasttext(self, model=None):

        if model is None:
            model = self.gensim


    def load_tokenized(self, fname=None):

        if fname is None:
            fname = 'X_tokenized.npy'

        return np.load(os.path.join(self.cfg.DATA_DIR, fname))


    def load_embedded(self, fname=None):

        if fname is None:
            fname = f'X_embed{self.cfg.embed_dim}.npy'

        return np.load(os.path.join(self.cfg.DATA_DIR, fname))


    def load_idx2word(self, fname=None):

        if fname is None:
            fname = 'idx2word.yml'
        
        with open(os.path.join(self.cfg.DATA_DIR, fname), 'r') as y:
            self.idx2word = yaml.load(y, Loader=yaml.FullLoader)
            self.word2idx = {v: k for k, v in self.idx2word.items()}
            return self.idx2word

    
    def load_fasttext(self, fname=None):

        if fname is None:
            fname = f'{self.cfg.pre_embed}{self.cfg.embed_dim}.model'

        if self.cfg.pre_embed == 'fasttext':
            self.gensim = FastText.load(os.path.join(self.cfg.DATA_DIR, fname))

        elif self.cfg.pre_embed == 'word2vec':
            self.gensim = Word2Vec.load(os.path.join(self.cfg.DATA_DIR, fname))
        return self.gensim

    
    @logging_time
    def __build_letter_mapper(self): # DEPRECATED

        # LETTER MAPPER
        #   make letter mapper that has
        #   used_letter: idx
        #   only includes numbers / alphabets

        self._pad = '<pad>'
        self._unk = '<unk>'

        ascii_list = list(range(48, 58)) + list(range(65, 91)) + list(range(97, 123))
        letter_mapper = {
            chr(_ascii): i+2 for i, _ascii in enumerate(ascii_list)
        }
        letter_mapper[self._pad] = 0
        letter_mapper[self._unk] = 1

        self.letter_mapper = letter_mapper
        return self.letter_mapper


    @logging_time
    def __build_unkown_letterset(self, X=None): # DEPRECATED

        # BUILD UNKNOWN LETTERSET
        #   should be run after -
        #   1. load_raw_papers X_Raw
        #   2. build_letter_mapper

        # Make unknown letter set
        if X is None:
            X = self.X_raw

        letter_counter = {k: 0 for k in self.letter_mapper.keys()}
        self.unk_letterset = set()
        for paper in X:
            
            for letter in paper:
                
                try:
                    if letter_counter.get(letter) >= 0:
                        letter_counter[letter] += 1
                    
                except:
                    letter_counter[self._unk] += 1
                    self.unk_letterset.add(letter)

        self.unk_letterset.remove(' ')
        return self.unk_letterset

  