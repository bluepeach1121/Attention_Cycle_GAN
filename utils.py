import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from typing import List
import random

class ReplayBuffer:
    def __init__(
        self, 
        max_size: int = 50, 
        device: torch.device = torch.device('cpu'), 
        replace_prob: float = 0.5
    ) -> None:
        assert max_size > 0, "Empty buffer."
        assert 0.0 <= replace_prob <= 1.0, "replace_prob must be between 0 and 1."
        
        self.max_size: int = max_size
        self.data: List[torch.Tensor] = []
        self.device: torch.device = device
        self.replace_prob: float = replace_prob

    def push_and_pop(self, data: torch.Tensor) -> torch.Tensor:
        to_return: List[torch.Tensor] = []
        for element in data:
            element = element.unsqueeze(0).to(self.device)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if torch.rand(1).item() < self.replace_prob:
                    i = torch.randint(0, self.max_size, (1,)).item()
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return).detach()


class LambdaLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        n_epochs: int,
        offset: int,
        decay_start_epoch: int,
        last_epoch: int = -1
    ) -> None:
        if not isinstance(n_epochs, int) or n_epochs <= 0:
            raise ValueError("n_epochs must be a positive integer.")
        if not isinstance(decay_start_epoch, int) or decay_start_epoch < 0:
            raise ValueError("decay_start_epoch must be a non-negative integer.")
        if decay_start_epoch >= n_epochs:
            raise ValueError("decay_start_epoch must be less than n_epochs.")
        
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
        super(LambdaLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        current_epoch = self.last_epoch
        lr_factor = 1.0 - max(
            0, 
            current_epoch + self.offset - self.decay_start_epoch
        ) / (self.n_epochs - self.decay_start_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]
