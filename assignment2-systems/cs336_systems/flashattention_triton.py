import torch
import triton
import triton.language as tl
import math

@triton.jit
def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        # Add IS_CAUSAL as a constexpr
        IS_CAUSAL: tl.constexpr,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    # 1. Block Pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + pid_b * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(pid_m * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + pid_b * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + pid_b * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + pid_b * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(pid_m * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    # L pointer logic
    offs_m = pid_m * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    L_ptr_base = L_ptr + pid_b * stride_lb + pid_m * Q_TILE_SIZE * stride_lq

    # 2. Accumulators
    m_i = tl.full([Q_TILE_SIZE], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    acc = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)

    # Load Q
    q = tl.load(Q_block_ptr, boundary_check=(0, 1))

    # 3. Loop over Key Tiles
    for k_start in range(0, N_KEYS, K_TILE_SIZE):

        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        v = tl.load(V_block_ptr, boundary_check=(0, 1))

        # Score: Q @ K.T
        qk = tl.dot(q, tl.trans(k))
        qk *= scale

        # --- CAUSAL MASKING LOGIC ---
        if IS_CAUSAL:
            # Current Key indices: [k_start, k_start + 1, ...]
            offs_n = k_start + tl.arange(0, K_TILE_SIZE)

            # Mask where Key Index (n) > Query Index (m)
            # Broadcasting: offs_n[None, :] vs offs_m[:, None]
            mask = offs_n[None, :] > offs_m[:, None]

            # Apply -inf to masked positions
            qk = tl.where(mask, float("-inf"), qk)
        # ----------------------------

        # Online Softmax
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        l_i = l_i * alpha + tl.sum(p, 1)

        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)

        m_i = m_i_new

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # 4. Finalize
    acc = acc / l_i[:, None]
    l_final = m_i + tl.log(l_i)

    # 5. Store
    tl.store(O_block_ptr, acc.to(Q_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_ptr_base + tl.arange(0, Q_TILE_SIZE) * stride_lq, l_final, mask=offs_m < N_QUERIES)

@torch.compile
def flash_attn_backward_logic(Q, K, V, O, L, dO, is_causal, scale):
    """
    Backward pass logic using PyTorch + torch.compile.
    Recomputes P and S to save memory (Activation Recomputation).

    Args:
        Q, K, V, O, dO: (Batch, Heads, Seq, Dim)
        L: (Batch, Heads, Seq) - LogSumExp from forward pass
    """
    # 1. Compute D (Delta)
    # Eq: D_i = sum(dO_i * O_i) over dimension d
    # Shape: (B, H, Seq, 1)
    D = torch.sum(dO * O, dim=-1, keepdim=True)

    # 2. Recompute Attention Matrix S
    # Shape: (B, H, Seq, Seq)
    # S_ij = scale * (Q_i . K_j^T)
    S = torch.matmul(Q, K.transpose(-1, -2)) * scale

    # Apply Causal Mask if needed (to ensure P is correct)
    if is_causal:
        seq_len = Q.shape[2]
        # Create mask: True for lower triangle (keep), False for upper (mask)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool))
        S = torch.where(mask, S, float("-inf"))

    # 3. Recompute Softmax Probabilities P
    # P_ij = exp(S_ij - L_i)
    # L shape needs broadcasting: (B, H, Seq, 1)
    P = torch.exp(S - L.unsqueeze(-1))

    # 4. Compute dV
    # dV_j = sum_i (P_ij * dO_i)  => dV = P^T @ dO
    dV = torch.matmul(P.transpose(-1, -2), dO)

    # 5. Compute dP
    # dP_ij = dO_i . V_j  => dP = dO @ V^T
    dP = torch.matmul(dO, V.transpose(-1, -2))

    # 6. Compute dS
    # dS_ij = P_ij * (dP_ij - D_i)
    dS = P * (dP - D)

    # Scale dS because S = QK^T * scale
    # d(QK^T) = dS * scale
    dS_scaled = dS * scale

    # 7. Compute dQ
    # dQ = dS_scaled @ K
    dQ = torch.matmul(dS_scaled, K)

    # 8. Compute dK
    # dK = dS_scaled^T @ Q
    dK = torch.matmul(dS_scaled.transpose(-1, -2), Q)

    return dQ, dK, dV


class FlashAttentionTritonFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        squeeze_head = False
        if Q.dim() == 3:
            squeeze_head = True
            Q = Q.unsqueeze(1)
            K = K.unsqueeze(1)
            V = V.unsqueeze(1)

        B, H, N_Q, D = Q.shape
        _, _, N_K, _ = K.shape

        O = torch.empty_like(Q)
        L = torch.empty((B, H, N_Q), device=Q.device, dtype=torch.float32)

        # Tiles: 32x32 worked for you on 3090.
        # You can try 64x64 if memory allows, but 32 is safe.
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32
        scale = 1.0 / math.sqrt(D)

        grid = (triton.cdiv(N_Q, Q_TILE_SIZE), B * H)

        Q_flat = Q.view(B * H, N_Q, D)
        K_flat = K.view(B * H, N_K, D)
        V_flat = V.view(B * H, N_K, D)
        O_flat = O.view(B * H, N_Q, D)
        L_flat = L.view(B * H, N_Q)

        flash_fwd_kernel[grid](
            Q_flat, K_flat, V_flat,
            O_flat, L_flat,
            Q_flat.stride(0), Q_flat.stride(1), Q_flat.stride(2),
            K_flat.stride(0), K_flat.stride(1), K_flat.stride(2),
            V_flat.stride(0), V_flat.stride(1), V_flat.stride(2),
            O_flat.stride(0), O_flat.stride(1), O_flat.stride(2),
            L_flat.stride(0), L_flat.stride(1),
            N_QUERIES=N_Q,
            N_KEYS=N_K,
            scale=scale,
            # PASS THE CAUSAL FLAG
            IS_CAUSAL=is_causal,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
        )

        if squeeze_head:
            Q = Q.squeeze(1)
            K = K.squeeze(1)
            V = V.squeeze(1)
            O = O.squeeze(1)
            L = L.squeeze(1)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.squeeze_head = squeeze_head
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, dO):
        # Copy the EXACT same backward logic from FlashAttentionTritonFunc
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        # Check dimensionality on the SAVED tensor before modifying it
        is_3d = (Q.ndim == 3)

        # Handle 3D inputs if needed
        if is_3d:
            dO = dO.unsqueeze(1)
            Q = Q.unsqueeze(1)
            K = K.unsqueeze(1)
            V = V.unsqueeze(1)
            O = O.unsqueeze(1)
            L = L.unsqueeze(1)

        scale = 1.0 / math.sqrt(Q.size(-1))

        # Reuse the compiled logic
        dQ, dK, dV = flash_attn_backward_logic(Q, K, V, O, L, dO, is_causal, scale)

        # Use the flag we saved earlier to decide whether to squeeze
        if is_3d:
            dQ = dQ.squeeze(1)
            dK = dK.squeeze(1)
            dV = dV.squeeze(1)

        return dQ, dK, dV, None