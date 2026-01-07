import torch
from cs336_basics_old.optimizer import SGD

# Initialize parameters
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))

# Initialize optimizer
opt = SGD([weights], lr=1e3)

# Training loop
for t in range(10):
    opt.zero_grad()                 # Reset the gradients for all learnable parameters.
    loss = (weights ** 2).mean()    # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward()                 # Run backward pass, which computes gradients.
    opt.step()                      # Run optimizer step.
