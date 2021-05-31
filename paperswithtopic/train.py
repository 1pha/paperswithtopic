from .config import load_config
from .preprocess import Preprocess
from .dataloader import get_dataloader
from .model import BERTClassification

def train(cfg):

    preprocess = Preprocess(cfg=cfg)
    if cfg.use_saved:
        X = preprocess.load_processed()
        y = preprocess.label    
        word_mapper = preprocess.load_word_mapper()
        
    else:
        X, y = preprocess.pp_pipeline()
        word_mapper = preprocess.word_mapper
        
    cfg.vocab_size = len(word_mapper)
    train_dataloader = get_dataloader(cfg, X, y)


if __name__=="__main__":

    cfg = load_config()
    cfg.use_saved = True
    train(cfg)