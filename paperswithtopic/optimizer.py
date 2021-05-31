from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup


def get_optimizer(model, cfg):

    lr = cfg.learning_rate
    weight_decay = cfg.weight_decay

    if cfg.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif cfg.optimizer == 'adamW':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 모든 parameter들의 grad값을 0으로 초기화
    optimizer.zero_grad()
    
    return optimizer


def get_scheduler(optimizer, args):

    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, mode='max', verbose=True)
        
    elif args.scheduler == 'linear_warmup':
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=args.total_steps)
    return scheduler

