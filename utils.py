import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from typing import List
import random

class ReplayBuffer:
    """
    A buffer that stores previously generated samples and provides a mechanism
    to reuse them during training to stabilize GAN training.

    Args:
        max_size (int): Maximum number of samples to store in the buffer.
        device (torch.device): Device where the tensors will be stored ('cpu' or 'cuda').
        replace_prob (float): Probability of replacing an old sample with a new one.
    """
    def __init__(
        self, 
        max_size: int = 50, 
        device: torch.device = torch.device('cpu'), 
        replace_prob: float = 0.5
    ) -> None:
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        assert 0.0 <= replace_prob <= 1.0, "replace_prob must be between 0 and 1."
        
        self.max_size: int = max_size
        self.data: List[torch.Tensor] = []
        self.device: torch.device = device
        self.replace_prob: float = replace_prob

    def push_and_pop(self, data: torch.Tensor) -> torch.Tensor:
        """
        Adds new samples to the buffer and returns a batch of samples.

        Args:
            data (torch.Tensor): A batch of new samples.

        Returns:
            torch.Tensor: A batch of samples from the buffer.
        """
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
    """
    Linearly decays the learning rate after a specified epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        n_epochs (int): Total number of epochs for training.
        offset (int): Starting epoch offset.
        decay_start_epoch (int): Epoch to start decaying the learning rate.
        last_epoch (int): The index of last epoch. Default: -1.
    """
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
        """
        Computes the learning rate factor for the current epoch.

        Returns:
            List[float]: A list of learning rates for each parameter group.
        """
        current_epoch = self.last_epoch
        lr_factor = 1.0 - max(
            0, 
            current_epoch + self.offset - self.decay_start_epoch
        ) / (self.n_epochs - self.decay_start_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]
