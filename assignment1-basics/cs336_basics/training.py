import torch
import numpy
import typing
import os

def get_batch(
        dataset: numpy.ndarray,
        batch_size: int,
        context_length: int,
        device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Samples a batch of input and target sequences from the dataset.

    Args:
        dataset: 1D numpy array of token IDs (supports np.memmap).
        batch_size: Number of sequences to sample.
        context_length: Length of each sequence.
        device: PyTorch device string (e.g. 'cpu', 'cuda:0').

    Returns:
        x: (batch_size, context_length) Input tokens
        y: (batch_size, context_length) Target tokens (shifted by 1)
    """
    # 1. Determine valid sampling range
    # We need to extract a sequence of length `context_length + 1`
    # (context_length for input x, and context_length for target y which is offset by 1)
    # The last valid starting index `i` must satisfy:
    # i + context_length < len(dataset)
    # (because y ends at i + context_length, which corresponds to index i + context_length)

    data_size = len(dataset)
    # high is exclusive in randint, so we set it to data_size - context_length
    high = data_size - context_length

    # 2. Sample random starting indices
    # We use torch.randint to generate batch_size indices
    ix = torch.randint(low=0, high=high, size=(batch_size,))

    # 3. Stack the batches
    # We iterate over the sampled indices and slice the numpy array.
    # Casting to int64 is crucial because PyTorch Embedding layers expect LongTensors.
    x_list = [torch.from_numpy((dataset[i : i + context_length]).astype(numpy.int64)) for i in ix]
    y_list = [torch.from_numpy((dataset[i + 1 : i + 1 + context_length]).astype(numpy.int64)) for i in ix]

    x = torch.stack(x_list)
    y = torch.stack(y_list)

    # 4. Move to device
    if device is not None:
        x = x.to(device)
        y = y.to(device)

    return x, y


def _unwrap_model(m: torch.nn.Module) -> torch.nn.Module:
    # torch.compile returns an OptimizedModule with attribute _orig_mod
    return m._orig_mod if hasattr(m, "_orig_mod") else m

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]]
) -> None:
    """
    Saves the model state, optimizer state, and iteration number to a file.
    """
    base_model = _unwrap_model(model)
    checkpoint_state = {
        "model": base_model.state_dict(),      # 永远保存原始模型的 key
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    # Dump it to the output file/path
    torch.save(checkpoint_state, out)

def load_checkpoint(
        src: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]], # <--- SRC MUST BE FIRST
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
) -> int:
    """
    Loads state from a checkpoint file.
    """
    ckpt = torch.load(src, map_location="cpu")

    # 1) 取出模型 state_dict
    state = ckpt["model"]

    # 2) 如果 ckpt 的 key 带 _orig_mod.，把它去掉
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k[len("_orig_mod."):]: v for k, v in state.items()}

    # 3) 如果 model 是 compiled，也要 unwrap
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    target.load_state_dict(state)

    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"]