import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Any, Type, Dict, Iterable

class ShardedOptimizer(Optimizer):
    def __init__(self, params: Iterable, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        # Get Distributed Info
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Helper to assign ownership round-robin style
        self.global_param_counter = 0

        # Buffer to store groups destined for the internal optimizer
        self.local_param_groups = []

        # 1. Initialize Base Class
        # This calls self.add_param_group() internally, which we override below.
        super().__init__(params, defaults=kwargs)

        # 2. Create the Internal Optimizer (The Shard)
        # It only contains parameters owned by THIS rank.
        # This saves memory because state is only allocated for these params.
        if self.local_param_groups:
            self.optim = self.optimizer_cls(self.local_param_groups, **self.optimizer_kwargs)
        else:
            # Handle edge case where a rank owns nothing
            self.optim = self.optimizer_cls([], **self.optimizer_kwargs)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """
        Split the param group:
        1. All params go to super() (for iteration/broadcasting).
        2. Only owned params go to self.optim (for updating/state storage).
        """
        # A. Assign Ownership Tag to every parameter
        for p in param_group['params']:
            if not hasattr(p, '_owner_rank'):
                p._owner_rank = self.global_param_counter % self.world_size
                self.global_param_counter += 1

        # B. Add to Global List (Base Class)
        super().add_param_group(param_group)

        # C. Create Local Group (Filter for owned params)
        # Copy dictionary to keep other settings (lr, weight_decay, etc.)
        local_group = {k: v for k, v in param_group.items() if k != 'params'}

        # Filter: Keep only params I own
        local_group['params'] = [
            p for p in param_group['params']
            if p._owner_rank == self.rank
        ]

        # D. Store/Add to Internal Optimizer
        if hasattr(self, 'optim'):
            if local_group['params']:
                self.optim.add_param_group(local_group)
        else:
            # We are inside __init__, buffer it for later
            self.local_param_groups.append(local_group)

    def step(self, closure=None, **kwargs):
        """
        1. Update local shard.
        2. Broadcast new weights to the world.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # 1. Local Update
        # Only updates weights for params where p._owner_rank == self.rank
        self.optim.step(**kwargs)

        # 2. Synchronization (Broadcast)
        # We iterate over ALL global parameters
        for group in self.param_groups:
            for p in group['params']:
                # The owner sends the updated data. Everyone else receives it.
                # src=p._owner_rank ensures the correct direction of data flow.
                dist.broadcast(p.data, src=p._owner_rank)

        return loss

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)