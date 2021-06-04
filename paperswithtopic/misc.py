import os
import time
import random
from datetime import datetime as dt

import numpy as np

import torch


def get_today():

    td = dt.today()

    return str(td.year) + str(td.month).zfill(2) + str(td.day).zfill(2) + '-' \
        + str(td.hour).zfill(2) + str(td.minute).zfill(2)


def seed_everything(seed=42):
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)    
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

def logging_time(original_fn):

    def wrapper_fn(*args, **kwargs):

        start_time = time.time()
        result = original_fn(*args, **kwargs)
        fn_name = original_fn.__name__
        end = '' if fn_name == 'train' else '\n'
        print(f"[{fn_name}] {time.time() - start_time:.1f} sec ", end=end)
        
        return result

    return wrapper_fn