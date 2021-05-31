import os
import time
import yaml
import numpy as np
import pandas as pd
from gensim.models import FastText

from .config import load_config, edict2dict

def logging_time(original_fn):

    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print(f"[{original_fn.__name__}] {end_time-start_time:.1f} sec ")
        return result

    return wrapper_fn


class Preprocess:

    def __init__(self, cfg=None):

        if cfg is None:
            cfg = load_config()

        self.cfg = cfg


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
    def pp_pipeline(self, path=None, return_y=True):

        '''
        1. LOAD RAW PAPERS
        2. BUILD LETTER MAPPER
        3. BUILD UNKNOWN LETTERSET
        4. REMOVE UNKNOWNS
        5. BUILD WORD MAPPER
        6. TOKENIZE (WORD2IDX)
        '''

        # 0. LOAD DATAFRAME
        df = self.load_data(path)

        # 1. LOAD RAW PAPERS
        X = self.retrieve_raw_papers(df)

        # 2. BUILD LETTER MAPPER
        #   This will make self.letter_mapper
        self.build_letter_mapper()

        # 3. BUILD UNKNOWN LETTERSET
        self.build_unkown_letterset(X)

        X = self.remove_unknown(X)
        self.build_word_mapper(X)
        X = self.tokenize_papers(X, self.cfg.PAD)

        if not return_y:
            return X
        else:
            return X, self.label


    @logging_time
    def retrieve_raw_papers(self, df):

        if df is None:
            self.df = df

        self.paper2idx = {p: i for p, i in enumerate(df.index)}
        self.X_raw = list(map(str.lower, self.paper2idx.values()))
        print(f'There are {len(self.X_raw)} papers.')
    
        return self.X_raw


    @logging_time
    def build_letter_mapper(self):

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
    def build_unkown_letterset(self, X=None):

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


    @logging_time
    def remove_unknown(self, X=None):

        if X is None:
            X = self.X_raw

        def _remove(sentence):

            for _unk in self.unk_letterset:
                sentence = sentence.replace(_unk, '')
            return sentence

        self.X_filter = list(map(_remove, X))
        return self.X_filter


    @logging_time
    def build_word_mapper(self, X=None):

        if X is None:
            X = self.X_filter

        self.word_mapper = {0: self._pad, 1: self._unk}
        def _encode(paper, idx):
            is_valid = lambda l: l in self.letter_mapper.keys()
            
            for word in list(filter(lambda x: x, paper.split(' '))):
                
                if word in self.word_mapper.values():
                    pass
                
                elif all(filter(is_valid, word)):
                    self.word_mapper[idx] = word
                    idx += 1
                    
            return idx

        idx = 2
        for paper in X:
            idx = _encode(paper, idx)

        return self.word_mapper


    @logging_time
    def tokenize_papers(self, X=None, pad=True, word_mapper=None):

        '''
        This will map word > index
        If pad is True, it will make every sequence to maximum sequence,
        so shorter sequences would have 0-pads in the back (post-padding).
        '''

        if X is None:
            X = self.X_filter

        if word_mapper is None:
            word_mapper = self.word_mapper

        word2idx = {v: k for k, v in self.word_mapper.items()}
        def _tokenize(paper):
            
            words = list(filter(lambda x: x, paper.split(' ')))
            SEQ_LEN = self.cfg.MAX_LEN if pad else len(words)
            tokens = [0 for _ in range(SEQ_LEN)]
            for idx in range(min(len(words), SEQ_LEN)):
                tokens[idx] = word2idx[words[idx]]
                
            return tokens

        self.X_tokenized = [_tokenize(paper) for paper in X]
        return self.X_tokenized


    @logging_time
    def fasttext_train(self, X=None, embed_dim=None, **kwargs):

        if X is None:
            X = self.X_tokenized

        if embed_dim is None:
            embed_dim = self.cfg.embed_dim

        X_fasttext = list(map(str.split, X))
        model = FastText(sentences=X_fasttext, vector_size=embed_dim, **kwargs)

        return model

    
    def save_data(self, X):

        np.save(os.path.join(self.cfg.DATA_DIR, 'X_tokenized.npy'), X)


    def save_word_mapper(self, fname=None):

        if fname is None:
            fname = 'word_mapper.yml'

        with open(os.path.join(self.cfg.DATA_DIR, fname), 'w') as y:
            yaml.dump(self.word_mapper, y)


    def load_processed(self, fname=None):

        if fname is None:
            fname = 'X_tokenized.npy'

        return np.load(os.path.join(self.cfg.DATA_DIR, fname))

    def load_word_mapper(self, fname=None):

        if fname is None:
            fname = 'word_mapper.yml'
        
        with open(os.path.join(self.cfg.DATA_DIR, fname), 'r') as y:
            self.word_mapper = yaml.load(y, Loader=yaml.FullLoader)
            return self.word_mapper
