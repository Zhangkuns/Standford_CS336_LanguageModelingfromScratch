import torch
import torch.distributed as dist

class DDPOverlapIndividual(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        # FIX 1: Must initialize the parent class first!
        super().__init__()

        self.module = module
        self.handles = []
        self.world_size = dist.get_world_size()

        # 1) Broadcast parameters and buffers ensures all ranks start with same weights
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)
            for b in self.module.buffers():
                dist.broadcast(b.data, src=0)

        # 2) Register hooks for async communication during backward
        for p in self.module.parameters():
            if p.requires_grad:
                # Use the helper to create a closure
                p.register_post_accumulate_grad_hook(self._make_hook())

    def _make_hook(self):
        # This hook runs immediately after the gradient for a specific param is calculated
        def hook_fn(param):
            if param.grad is None:
                return

            # FIX 2: Async All-Reduce on the gradient
            # Note: param.grad is the tensor we want to sync
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)

            # Note: register_post_accumulate_grad_hook should NOT return anything
        return hook_fn

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        # 等待所有 async all_reduce 被排队/完成
        for h in self.handles:
            h.wait()
        self.handles.clear()

        # 别忘了平均
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad.div_(self.world_size)

class Bucket:
    """
    Helper class to manage a single bucket of parameters.
    """
    def __init__(self, params: list[torch.nn.Parameter], index: int):
        self.index = index
        self.params = params
        self.count_ready = 0
        self.comm_handle = None

        # 1. Allocate a flat buffer for this bucket
        # We assume all params in a bucket have the same dtype/device (enforced during bucketing)
        total_numel = sum(p.numel() for p in params)
        dtype = params[0].dtype
        device = params[0].device

        self.buffer = torch.zeros(total_numel, dtype=dtype, device=device)

        # 2. Create views into the buffer for each parameter
        # This allows us to copy p.grad into the right spot of self.buffer easily
        self.param_views = {}
        offset = 0
        for p in params:
            # Create a view of the buffer that matches p's shape
            self.param_views[p] = self.buffer[offset : offset + p.numel()].view(p.shape)
            offset += p.numel()

    def reset(self):
        """Reset state for the next iteration."""
        self.count_ready = 0
        self.comm_handle = None
        # Note: We don't need to zero the buffer, we overwrite it with grads.

class DDPBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        # Convert MB to bytes (1 MB = 1024 * 1024 bytes)
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024

        # 1. Broadcast Init Weights
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)
            for b in self.module.buffers():
                dist.broadcast(b.data, src=0)

        # 2. Assign Parameters to Buckets (Reverse Order)
        self.buckets = []
        self.param_to_bucket = {}

        # We iterate in reverse because gradients are computed from Output -> Input
        trainable_params = [p for p in self.module.parameters() if p.requires_grad]
        reversed_params = list(reversed(trainable_params))

        current_bucket_params = []
        current_bucket_size = 0

        for p in reversed_params:
            # Check size (numel * element_size)
            p_bytes = p.numel() * p.element_size()

            # If adding this param exceeds bucket size, close current bucket
            # (unless current bucket is empty, then we must add at least one)
            if current_bucket_params and (current_bucket_size + p_bytes > self.bucket_size_bytes):
                self._create_bucket(current_bucket_params)
                current_bucket_params = []
                current_bucket_size = 0

            # Ideally we also check if dtype/device matches, but for this assignment
            # we assume consistent model dtype/device.
            current_bucket_params.append(p)
            current_bucket_size += p_bytes

        # Create final bucket
        if current_bucket_params:
            self._create_bucket(current_bucket_params)

        # 3. Register Hooks
        for p in trainable_params:
            p.register_post_accumulate_grad_hook(self._make_hook(p))

    def _create_bucket(self, params):
        bucket_idx = len(self.buckets)
        bucket = Bucket(params, bucket_idx)
        self.buckets.append(bucket)
        # Map params to this bucket for easy lookup in hooks
        for p in params:
            self.param_to_bucket[p] = bucket

    def _make_hook(self, param):
        # We need to know WHICH bucket this param belongs to.
        # We capture 'bucket' via the closure, but we look it up in init.
        # However, register_post_accumulate_grad_hook passes 'param' to function.
        # We can just look up self.param_to_bucket[param] inside.

        def hook_fn(p):
            if p.grad is None:
                return

            bucket = self.param_to_bucket[p]

            # A. Copy Gradient to Bucket Buffer
            # We copy p.grad into the pre-allocated contiguous buffer view
            bucket.param_views[p].copy_(p.grad)

            # B. Increment Counter
            bucket.count_ready += 1

            # C. If Bucket Full, Launch Async All-Reduce
            if bucket.count_ready == len(bucket.params):
                # Trigger All-Reduce on the flattened buffer
                bucket.comm_handle = dist.all_reduce(
                    bucket.buffer,
                    op=dist.ReduceOp.SUM,
                    async_op=True
                )

        return hook_fn

    def forward(self, *args, **kwargs):
        # Reset buckets before forward pass (prepare counters)
        for bucket in self.buckets:
            bucket.reset()
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        # Iterate through buckets in creation order (Reverse of model)
        # Wait for comms and copy back

        for bucket in self.buckets:
            # 1. Wait for communication
            if bucket.comm_handle is not None:
                bucket.comm_handle.wait()

            # 2. Average and Copy Back
            # We can divide the whole buffer at once for speed
            bucket.buffer.div_(self.world_size)

            # Copy data back from buffer views to p.grad
            for p in bucket.params:
                if p.grad is not None:
                    p.grad.copy_(bucket.param_views[p])
