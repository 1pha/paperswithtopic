import wandb

from .config import load_config
from .run import run


if __name__=="__main__":

    cfg = load_config()
    cfg.use_saved = True

    wandb.login()
    wandb.init(
        project_name='paperswithtopic',
        config=vars(cfg),
        name='Test run'
    )
    
    run(cfg)