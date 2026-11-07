"""Minimum-loss checkpoint selection, including an untrained baseline."""
from dataclasses import dataclass
import math

@dataclass
class EarlyStopping:
    patience: int = 2
    best: float = float('inf')
    best_epoch: int = 0
    stale: int = 0

    def observe(self, epoch: int, loss: float) -> bool:
        if not math.isfinite(loss):
            raise ValueError('Validation loss must be finite')
        improved = loss < self.best
        if improved:
            self.best, self.best_epoch, self.stale = loss, epoch, 0
        elif epoch:
            self.stale += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.stale >= self.patience
