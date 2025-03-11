import math
from torch.optim import lr_scheduler

def fastReID_lr_lambda(iteration):
    num_iterations = 18000
    warmup = 2000
    decay_start = 9000
    decay_iters = num_iterations - decay_start
    if iteration < warmup:
        return 0.01 + 0.99 * (iteration / warmup)
    elif iteration < decay_start:
        return 1.0
    else:
        decay_step = min(iteration - decay_start, decay_iters)
        return 0.5 * (1 + math.cos(math.pi * decay_step/decay_iters))

def build_lr_scheduler(optimizer, warmup=0, epochs=60, type_scheduler='cosine'):
    
    if type_scheduler == 'cosine':
        main_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs-warmup, eta_min=1e-5
        )
    else:
        main_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    if warmup == 0:
        return main_scheduler

    warmup_scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (epoch + 1)/warmup
    )
    custom_scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup]
    )
    return custom_scheduler
