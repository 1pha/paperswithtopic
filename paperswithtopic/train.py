from .config import load_config
from .preprocess import Preprocess
from .dataloader import get_dataloader



if __name__=="__main__":

    cfg = load_config()
    cfg.use_saved = True