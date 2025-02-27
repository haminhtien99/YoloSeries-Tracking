import math


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