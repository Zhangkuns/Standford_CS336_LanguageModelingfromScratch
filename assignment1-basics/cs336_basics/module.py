import math
import torch
import torch.nn as nn
import einops

class LinearModule(nn.Module):
    def __init__(self, d_in: int, d_out: int, device=None, dtype=None):
        """
        Construct a linear transformation module without bias.

        Args:
            d_in: size of each input sample
            d_out: size of each output sample
            device: the device on which to allocate the parameter
            dtype: the dtype of the parameter
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        # Constraint: Store parameter as W (shape: [in, out]), NOT W^T.
        # This is different from standard PyTorch nn.Linear which uses [out, in].
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty((d_out, d_in), **factory_kwargs))

        # Initialize weights immediately
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights using Truncated Normal distribution.
        Formula: N(0, sigma^2) truncated at [-3sigma, 3sigma]
        where sigma = sqrt(2 / (fan_in + fan_out))
        """
        # Calculate standard deviation (sigma)
        # d_in + d_out
        fan_sum = self.d_in + self.d_out
        sigma = math.sqrt(2.0 / fan_sum)

        # Calculate truncation bounds
        bound = 3.0 * sigma

        # Apply initialization
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=sigma,
            a=-bound,
            b=bound
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation using einsum.
        x shape: (batch..., d_in)
        weight shape: (d_out, d_in)
        output shape: (batch..., d_out)
        """
        # "..." represents any number of batch dimensions (e.g., Batch, Sequence Length)
        return einops.einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        Construct an embedding module.

        Args:
            num_embeddings: Size of the vocabulary
            embedding_dim: Dimension of the embedding vectors
            device: The device on which to allocate the parameter
            dtype: The dtype of the parameter
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize the embedding matrix as a Parameter
        # Shape: (vocab_size, d_model)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights: N(mu=0, sigma^2=1) truncated at [-3, 3].
        """
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.

        Args:
            token_ids: Tensor of shape (batch_size, sequence_length) containing integer IDs.

        Returns:
            Tensor of shape (batch_size, sequence_length, embedding_dim)
        """
        # PyTorch supports direct indexing with tensors to select rows.
        # This effectively performs the lookup without using nn.functional.embedding.
        return self.weight[token_ids]

# class RMSNorm(nn.Module):
#     def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
#         super().__init__()
#         # Use the official PyTorch implementation
#         self.norm = torch.nn.RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # The official implementation handles upcasting internally for stability
#         return self.norm(x)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Upcast
        in_dtype = x.dtype
        x_fp32 = x.to(torch.float32)

        # 2. Calculate RMS (Standard torch.mean is safer/faster here than einops.reduce)
        # keepdim=True preserves the last dim as 1 for broadcasting
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(variance + self.eps) # rsqrt is 1/sqrt(x), faster

        # 3. Normalize
        x_norm = x_fp32 * rms

        # 4. Scale
        # Use standard broadcasting (*) instead of einsum
        # x_norm: (Batch, Seq, D) * weight: (D) -> Broadcasts automatically
        output = x_norm * self.weight

        return output.to(in_dtype)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """
        SwiGLU Feed-Forward Network.
        Formula: FFN(x) = W2(SiLU(W1x) * W3x)

        Args:
            d_model: Input/Output dimension.
            d_ff: Hidden dimension.
        """
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        # W1 (Gate) and W3 (Value) project from d_ff -> d_model
        # Stored as (in, out) per assignment convention
        self.w1 = nn.Parameter(torch.empty((d_ff, d_model), **factory_kwargs))
        self.w3 = nn.Parameter(torch.empty((d_ff, d_model), **factory_kwargs))

        # W2 (Output) projects from d_model -> d_ff
        self.w2 = nn.Parameter(torch.empty((d_model, d_ff), **factory_kwargs))

        self.reset_parameters(d_model, d_ff)

    def reset_parameters(self, d_model, d_ff):
        # Initialization logic (sigma = sqrt(2 / (fan_in + fan_out)))

        # W1 & W3: d_model -> d_ff
        std_in = (2.0 / (d_model + d_ff)) ** 0.5
        for w in [self.w1, self.w3]:
            torch.nn.init.trunc_normal_(w, mean=0.0, std=std_in, a=-3*std_in, b=3*std_in)

        # W2: d_ff -> d_model
        std_out = (2.0 / (d_ff + d_model)) ** 0.5
        torch.nn.init.trunc_normal_(self.w2, mean=0.0, std=std_out, a=-3*std_out, b=3*std_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU FFN.
        x: (..., d_model)
        """
        # 1. Project to hidden dimension (Gate W1 and Value W3)
        # W2(SiLU(W1x) ⊙ W3x)
        # Using einops to handle arbitrary batch dimensions ("...")
        w1x = einops.einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        w3x = einops.einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")

        # 2. Apply SwiGLU activation
        # SiLU(W1x) = x * sigmoid(x).
        act = w1x * torch.sigmoid(w1x)

        # GLU(x, W1, W3) = σ(W1x) ⊙ W3x = SiLU(W1x) ⊙ W3x
        h = act * w3x

        # W2(SiLU(W1x) ⊙ W3x)
        output = einops.einsum(h, self.w2, "... d_ff, d_model d_ff -> ... d_model")

        return output


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Rotary Positional Embedding Module.

        Args:
            theta: Base frequency.
            d_k: Dimension of the key/query vectors (must be even).
            max_seq_len: Maximum sequence length to precompute embeddings for.
        """
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for rotary embeddings."

        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Precompute the sinusoidal frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        position_ids = torch.arange(0, max_seq_len, device=device).float()
        sinusoid_inp = torch.einsum('i , j -> i j', position_ids, inv_freq)

        # Compute sin and cos matrices
        self.register_buffer('sin_emb', torch.sin(sinusoid_inp), persistent=False)
        self.register_buffer('cos_emb', torch.cos(sinusoid_inp), persistent=False)

    def forward(self, x: torch.Tensor, pos_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embeddings to input tensor x.

        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            pos_ids: Tensor of shape (..., seq_len) containing position indices.

        Returns:
            Tensor of the same shape as x with rotary embeddings applied.
        """
        # Get sin and cos embeddings for the given position IDs
        sin_emb = self.sin_emb[pos_ids]  # Shape: (..., seq_len, d_k/2)
        cos_emb = self.cos_emb[pos_ids]  # Shape: (..., seq_len, d_k/2)

        # If input x is 4D (Batch, Heads, Seq, Dim), we must unsqueeze the embeddings
        # at dim 1 so they broadcast across heads: (Batch, 1, Seq, Dim/2)
        if x.ndim == 4:
            sin_emb = sin_emb.unsqueeze(1)
            cos_emb = cos_emb.unsqueeze(1)

        # Split x into two halves
        x_even = x[..., 0::2]     # (..., d_k/2)
        x_odd  = x[..., 1::2]     # (..., d_k/2)


       # Apply rotary transformation
        x_rotated_even = x_even * cos_emb - x_odd * sin_emb
        x_rotated_odd  = x_even * sin_emb + x_odd * cos_emb

        x_pair = torch.stack((x_rotated_even, x_rotated_odd), dim=-2)  # (..., 2, d2)
        x_rotated = einops.rearrange(x_pair, "... two d2 -> ... (d2 two)")
        return x_rotated

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Applies the Softmax function to an n-dimensional input Tensor.
    Uses the "subtract max" trick for numerical stability.

    Formula: exp(x_i - x_max) / sum(exp(x_j - x_max))
    """
    # 1. Find the maximum value along the specific dimension.
    # keepdim=True is critical so we can broadcast the subtraction later.
    # torch.max returns (values, indices), we only need values [0].
    x_max = torch.max(x, dim=dim, keepdim=True)[0]

    # 2. Subtract the max for numerical stability.
    # This ensures the largest value exponentiated is exp(0)=1, preventing overflow (infinity).
    # Broadcasting handles the shapes automatically.
    x_shifted = x - x_max

    # 3. Compute exponentials
    x_exp = torch.exp(x_shifted)

    # 4. Normalize
    # Sum along the same dimension to get the denominator.
    x_sum = einops.reduce(x_exp, '... n -> ... 1', 'sum')

    return x_exp / x_sum

def scaled_dot_product_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Computes scaled dot-product attention.

    Args:
        q (Float[Tensor, " ... queries d_k"]): Query tensor
        k (Float[Tensor, " ... keys d_k"]): Key tensor
        v (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
              True = Attend, False = Mask out.

    Returns:
        Output tensor of shape (..., seq_len, d_v)
    """
    # 1. Calculate dimension d_k
    d_k = q.size(-1)
    d_v = v.size(-1)

    # 2. Compute Scores: Q K^T / sqrt(d_k)
    # Use einsum to handle arbitrary batch dimensions ("...")
    # and perform the dot product along the embedding dimension 'd'.
    # Shape: (..., seq_len_q, seq_len_k)
    scores = einops.einsum(q, k, "... queries d_k, ... keys d_k -> ... queries keys")

    scores = scores / (d_k ** 0.5)

    # 3. Apply Masking
    if mask is not None:
        # We need to fill positions where mask is False with -inf.

        # Create a large negative number (simulating -infinity for softmax)
        # We use -1e4 or -inf. -float('inf') is standard.
        neg_inf = float('-inf')

        # We use masked_fill. Note: mask is True for "Keep", False for "Mask".
        # masked_fill expects True for "Replace".
        # So we want to replace where mask is False (i.e., replace where ~mask is True).
        scores = scores.masked_fill(~mask, neg_inf)

    # 4. Apply Softmax
    # Apply along the last dimension (the key sequence length dimension)
    attn_weights = softmax(scores, dim=-1)

    # 5. Compute Output: Weights * V
    # Shape: (..., seq_len_q, d_v)
    output = einops.einsum(attn_weights, v, "... queries d_k, ... d_k d_v -> ... queries d_v")

    return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            max_seq_len: int = 2048,
            rope_theta: float = 10000.0,
            use_rope: bool = True,  # <--- New flag
            device=None,
            dtype=None
    ):
        super().__init__()
        self.d_in = d_model
        self.d_out = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.use_rope = use_rope  # <--- Store flag

        if self.d_k * num_heads != d_model:
            raise ValueError(f"d_model {d_model} must be divisible by num_heads {num_heads}")

        factory_kwargs = {'device': device, 'dtype': dtype}

        # 1. Projections
        self.q_proj = nn.Parameter(torch.empty((num_heads * self.d_k, self.d_in), **factory_kwargs))
        self.k_proj = nn.Parameter(torch.empty((num_heads * self.d_k, self.d_in), **factory_kwargs))
        self.v_proj = nn.Parameter(torch.empty((num_heads * self.d_v, self.d_in), **factory_kwargs))
        self.o_proj = nn.Parameter(torch.empty((d_model, self.d_out), **factory_kwargs))

        # 2. RoPE (Only init if used)
        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=rope_theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device
            )
        else:
            self.rope = None

        self.reset_parameters()

    def reset_parameters(self):
        std = (1.0 / self.d_in) ** 0.5
        for w in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            torch.nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        x: (Batch, Seq, D_Model)
        token_positions: Optional (Batch, Seq) integer tensor
        """
        batch_size, seq_len, _ = x.shape

        # 1. Project
        q = einops.einsum(x, self.q_proj, "... sequence_length d_in, hd_k d_in -> ... sequence_length hd_k")
        k = einops.einsum(x, self.k_proj, "... sequence_length d_in, hd_k d_in -> ... sequence_length hd_k")
        v = einops.einsum(x, self.v_proj, "... sequence_length d_in, hd_v d_in -> ... sequence_length hd_v")


        # 2. Split Heads
        q = einops.rearrange(q, '... sequence_length (h d_k) -> ... h sequence_length d_k', h=self.num_heads)
        k = einops.rearrange(k, '... sequence_length (h d_k) -> ... h sequence_length d_k', h=self.num_heads)
        v = einops.rearrange(v, '... sequence_length (h d_v) -> ... h sequence_length d_v', h=self.num_heads)

        # 3. Apply RoPE (Optional)
        if self.use_rope:
            if token_positions is None:
                # Default to 0, 1, 2... if not provided
                # shape: (1, seq_len) -> expanded to (batch, seq_len)
                positions = einops.repeat(torch.arange(seq_len, device=x.device).unsqueeze(0), '1 s -> b s', b=batch_size)
            else:
                positions = token_positions

            q = self.rope(q, positions)
            k = self.rope(k, positions)

        # 4. Masking (Causal)
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))

        # 5. Attention
        attn_output = scaled_dot_product_attention(q, k, v, mask=mask)

        # 6. Concatenate
        attn_output = einops.rearrange(attn_output, '... h sequence_length d_v -> ... sequence_length (h d_v)')

        # 7. Output Project
        output = einops.einsum(attn_output, self.o_proj, " ... sequence_length d_model, d_out d_model-> ... sequence_length d_out")

        return output


class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int = 2048,
            rope_theta: float = 10000.0,
            device=None,
            dtype=None
    ):
        """
        A standard Pre-Norm Transformer Block.
        Structure:
            x = x + Attention(RMSNorm(x))
            x = x + FeedForward(RMSNorm(x))
        """
        super().__init__()

        # 1. Attention Sublayer
        self.rms_1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            use_rope=True,
            device=device,
            dtype=dtype
        )

        # 2. Feed-Forward Sublayer
        self.rms_2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: (Batch, Seq_Len, D_Model)
        """
        # Sublayer 1: Norm -> Attention -> Add (Residual)
        # Note: We pass the normalized input to attention, but add the *original* x
        h = self.rms_1(x)
        h = self.attn(h)
        x = x + h

        # Sublayer 2: Norm -> FFN -> Add (Residual)
        h = self.rms_2(x)
        h = self.ffn(h)
        x = x + h

        return x


class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float = 10000.0,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        # 1. Token Embeddings
        self.token_embeddings = Embedding(vocab_size, d_model, **factory_kwargs)

        # 2. Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                rope_theta=rope_theta,
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)
        ])

        # 3. Final Layer Norm
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

        # 4. LM Head (Output Projection) using LinearModule
        # Input: d_model, Output: vocab_size
        # LinearModule initializes itself, so we don't need manual init here.
        self.lm_head = LinearModule(d_in=d_model, d_out=vocab_size, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Token IDs of shape (Batch, Seq_Len)
        Returns: Logits of shape (Batch, Seq_Len, Vocab_Size)
        """
        # 1. Embed
        x = self.token_embeddings(x)

        # 2. Layers
        for layer in self.layers:
            x = layer(x)

        # 3. Norm
        x = self.ln_final(x)

        # 4. LM Head
        # LinearModule.forward handles the einsum/matmul
        logits = self.lm_head(x)

        return logits