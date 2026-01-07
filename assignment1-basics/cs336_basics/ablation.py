import math
import torch
import torch.nn as nn
import einops
from cs336_basics.module import RMSNorm,MultiHeadSelfAttention, FeedForward, Embedding, LinearModule, scaled_dot_product_attention, RotaryPositionalEmbedding

class TransformerBlockWithoutLayerNorm(nn.Module):
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
        h = x
        h = self.attn(h)
        x = x + h

        # Sublayer 2: Norm -> FFN -> Add (Residual)
        h = x
        h = self.ffn(h)
        x = x + h

        return x


class TransformerLMWithoutLayerNorm(nn.Module):
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
            TransformerBlockWithoutLayerNorm(
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


class TransformerBlockPostNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int = 2048,
            rope_theta: float = 10000.0,
            device=None,
            dtype=None,
    ):
        """
        Post-Norm Transformer block (your current computation order):
            h = Attn(x)
            z = RMSNorm(x + h)
            h = FFN(z)
            y = RMSNorm(z + h)
        """
        super().__init__()

        # Attention sublayer
        self.rms_1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            use_rope=True,
            device=device,
            dtype=dtype,
        )

        # Feed-forward sublayer
        self.rms_2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        # Sublayer 1: Attn -> Add -> Norm (post-norm)
        attn_out = self.attn(x)
        z = self.rms_1(x + attn_out)

        # Sublayer 2: FFN -> Add -> Norm (post-norm)
        ffn_out = self.ffn(z)
        y = self.rms_2(z + ffn_out)

        return y


class TransformerLMPostNorm(nn.Module):
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
            TransformerBlockPostNorm(
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

class MultiHeadSelfAttentionNoRope(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            max_seq_len: int = 2048,
            rope_theta: float = 10000.0,
            use_rope: bool = False,  # <--- New flag
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
        # if self.use_rope:
        #     self.rope = RotaryPositionalEmbedding(
        #         theta=rope_theta,
        #         d_k=self.d_k,
        #         max_seq_len=max_seq_len,
        #         device=device
        #     )
        # else:
        #     self.rope = None

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

        # 4. Masking (Causal)
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))

        # 5. Attention
        attn_output = scaled_dot_product_attention(q, k, v, mask=mask)

        # 6. Concatenate
        attn_output = einops.rearrange(attn_output, '... h sequence_length d_v -> ... sequence_length (h d_v)')

        # 7. Output Project
        output = einops.einsum(attn_output, self.o_proj, " ... sequence_length d_model, d_out d_model-> ... sequence_length d_out")

        return output

class TransformerBlockNoRope(nn.Module):
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
        self.attn = MultiHeadSelfAttentionNoRope(
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


class TransformerLMNoRope(nn.Module):
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
            TransformerBlockNoRope(
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



class FeedForwardSiLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """
        SiLU Feed-Forward Network (no gating).
        Formula: FFN(x) = W2(SiLU(W1x))

        Args:
            d_model: Input/Output dimension.
            d_ff: Hidden dimension (for this ablation, set d_ff = 4 * d_model).
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        # W1: project d_model -> d_ff
        # Stored as (out, in) = (d_ff, d_model) per your convention
        self.w1 = nn.Parameter(torch.empty((d_ff, d_model), **factory_kwargs))

        # W2: project d_ff -> d_model
        # Stored as (out, in) = (d_model, d_ff)
        self.w2 = nn.Parameter(torch.empty((d_model, d_ff), **factory_kwargs))

        self.reset_parameters(d_model, d_ff)

    def reset_parameters(self, d_model: int, d_ff: int) -> None:
        # W1: d_model -> d_ff
        std_in = (2.0 / (d_model + d_ff)) ** 0.5
        torch.nn.init.trunc_normal_(self.w1, mean=0.0, std=std_in, a=-3 * std_in, b=3 * std_in)

        # W2: d_ff -> d_model
        std_out = (2.0 / (d_ff + d_model)) ** 0.5
        torch.nn.init.trunc_normal_(self.w2, mean=0.0, std=std_out, a=-3 * std_out, b=3 * std_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d_model)
        """
        # Hidden: (..., d_ff)
        h = einops.einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")

        # SiLU activation: x * sigmoid(x)
        h = h * torch.sigmoid(h)

        # Output: (..., d_model)
        out = einops.einsum(h, self.w2, "... d_ff, d_model d_ff -> ... d_model")
        return out



class TransformerBlockSiLU(nn.Module):
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
        self.ffn = FeedForwardSiLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

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


class TransformerLMSiLU(nn.Module):
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
            TransformerBlockSiLU(
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

class MultiHeadSelfAttentionAddGate(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            max_seq_len: int = 2048,
            rope_theta: float = 10000.0,
            use_rope: bool = True,
            device=None,
            dtype=None
    ):
        super().__init__()
        self.d_in = d_model
        self.d_out = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.use_rope = use_rope

        if self.d_k * num_heads != d_model:
            raise ValueError(f"d_model {d_model} must be divisible by num_heads {num_heads}")

        factory_kwargs = {'device': device, 'dtype': dtype}

        # 1. Projections
        # Standard Q, K, V (Out, In)
        self.q_proj = nn.Parameter(torch.empty((num_heads * self.d_k, self.d_in), **factory_kwargs))
        self.k_proj = nn.Parameter(torch.empty((num_heads * self.d_k, self.d_in), **factory_kwargs))
        self.v_proj = nn.Parameter(torch.empty((num_heads * self.d_v, self.d_in), **factory_kwargs))

        # --- NEW: Gate Projection ---
        # Projects Input -> (Num_Heads * Head_Dim)
        # This matches the "Elementwise G1" variant in the paper (Table 1, Row 5)
        self.gate_proj = nn.Parameter(torch.empty((num_heads * self.d_v, self.d_in), **factory_kwargs))

        self.o_proj = nn.Parameter(torch.empty((self.d_out, d_model), **factory_kwargs))

        # 2. RoPE
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
        for w in [self.q_proj, self.k_proj, self.v_proj, self.gate_proj, self.o_proj]:
            torch.nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 1. Project Q, K, V ... AND GATE
        # Weights are (Out, In). Contract dim 1.
        q = einops.einsum(x, self.q_proj, "... d_in, hd_k d_in -> ... hd_k")
        k = einops.einsum(x, self.k_proj, "... d_in, hd_k d_in -> ... hd_k")
        v = einops.einsum(x, self.v_proj, "... d_in, hd_v d_in -> ... hd_v")

        # Calculate Gate Logits from Input X (Query-Dependent)
        gate = einops.einsum(x, self.gate_proj, "... d_in, hd_v d_in -> ... hd_v")

        # 2. Split Heads (Include Gate in the split)
        q = einops.rearrange(q, '... seq (h d) -> ... h seq d', h=self.num_heads)
        k = einops.rearrange(k, '... seq (h d) -> ... h seq d', h=self.num_heads)
        v = einops.rearrange(v, '... seq (h d) -> ... h seq d', h=self.num_heads)

        # Split Gate: (Batch, Heads, Seq, Head_Dim)
        gate = einops.rearrange(gate, '... seq (h d) -> ... h seq d', h=self.num_heads)

        # 3. Apply RoPE
        if self.use_rope:
            if token_positions is None:
                positions = einops.repeat(torch.arange(seq_len, device=x.device).unsqueeze(0), '1 s -> b s', b=batch_size)
            else:
                positions = token_positions
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        # 4. Attention
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
        attn_output = scaled_dot_product_attention(q, k, v, mask=mask)

        # --- APPLY GATING (The $G_1$ Position) ---
        # 1. Non-linearity: Sigmoid
        # 2. Sparsity: Element-wise multiplication
        # Since 'gate' and 'attn_output' are both (Batch, Heads, Seq, Dim), this works perfectly.
        attn_output = attn_output * torch.sigmoid(gate)

        # 6. Concatenate
        attn_output = einops.rearrange(attn_output, '... h seq d -> ... seq (h d)')

        # 7. Output Project
        output = einops.einsum(attn_output, self.o_proj, " ... seq d_in, d_out d_in -> ... seq d_out")

        return output

class TransformerBlockAddGate(nn.Module):
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
        self.attn = MultiHeadSelfAttentionAddGate(
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


class TransformerLMAddGate(nn.Module):
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
            TransformerBlockAddGate(
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
