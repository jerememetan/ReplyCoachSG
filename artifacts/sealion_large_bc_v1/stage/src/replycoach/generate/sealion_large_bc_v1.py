"""Optimizer-update schedule, independent of microbatches and validation calls."""
import math

def lr_factor(update, total_updates, warmup_updates):
    """1-based upcoming optimizer update; warmup reaches peak, last update is zero."""
    if not 1 <= update <= total_updates or not 1 <= warmup_updates < total_updates:
        raise ValueError('Invalid scheduler position/budget')
    if update <= warmup_updates:
        return update / warmup_updates
    fraction = (update - warmup_updates) / (total_updates - warmup_updates)
    return .5 * (1 + math.cos(math.pi * fraction))
