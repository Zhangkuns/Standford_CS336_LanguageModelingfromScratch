import torch
import math

from cs336_systems.flashattention_triton import flash_attn_backward_logic


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        PyTorch reference implementation of FlashAttention-2 Forward Pass.
        """
        # Handle 3D inputs (Batch, Seq, Dim) by treating them as 1 Head
        is_3d = Q.ndim == 3
        if is_3d:
            Q_in = Q.unsqueeze(1)
            K_in = K.unsqueeze(1)
            V_in = V.unsqueeze(1)
        else:
            Q_in = Q
            K_in = K
            V_in = V

        # Shapes
        batch_size, num_heads, n_q, d_head = Q_in.shape
        _, _, n_k, _ = K_in.shape

        # Tile sizes
        BLOCK_M = 32
        BLOCK_N = 32

        # Initialize Output and LogSumExp tensors
        O = torch.zeros_like(Q_in)
        L = torch.zeros(batch_size, num_heads, n_q, device=Q.device, dtype=torch.float32)

        scale = 1.0 / math.sqrt(d_head)

        for b in range(batch_size):
            for h in range(num_heads):
                q_bh = Q_in[b, h]
                k_bh = K_in[b, h]
                v_bh = V_in[b, h]

                # Split Q into tiles (Outer loop)
                for i_start in range(0, n_q, BLOCK_M):
                    i_end = min(i_start + BLOCK_M, n_q)
                    q_i = q_bh[i_start:i_end]

                    # Initialize running statistics for online softmax
                    m_i = torch.full((i_end - i_start, 1), float('-inf'), device=Q.device, dtype=torch.float32)
                    l_i = torch.zeros((i_end - i_start, 1), device=Q.device, dtype=torch.float32)
                    o_i = torch.zeros_like(q_i)

                    # Split K, V into tiles (Inner loop)
                    for j_start in range(0, n_k, BLOCK_N):
                        j_end = min(j_start + BLOCK_N, n_k)
                        k_j = k_bh[j_start:j_end]
                        v_j = v_bh[j_start:j_end]

                        # 1. Compute Score Tile
                        s_ij = torch.matmul(q_i, k_j.transpose(-1, -2)) * scale

                        # 2. Update Running Max
                        m_ij, _ = torch.max(s_ij, dim=1, keepdim=True)
                        m_i_new = torch.max(m_i, m_ij)

                        # 3. Compute P_tilde (Unnormalized probs)
                        p_tilde = torch.exp(s_ij - m_i_new)

                        # 4. Correction Factor
                        alpha = torch.exp(m_i - m_i_new)

                        # 5. Update Running Denominator
                        l_i = l_i * alpha + torch.sum(p_tilde, dim=1, keepdim=True)

                        # 6. Update Output Accumulator
                        pv_j = torch.matmul(p_tilde, v_j)
                        o_i = o_i * alpha + pv_j

                        m_i = m_i_new

                    # Final Normalization for Tile i
                    o_final = o_i / l_i
                    l_final = m_i + torch.log(l_i)

                    O[b, h, i_start:i_end] = o_final
                    L[b, h, i_start:i_end] = l_final.squeeze(-1)

        # Restore original dimensions if input was 3D
        if is_3d:
            O = O.squeeze(1)
            L = L.squeeze(1)

        # Save for backward
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        # FIX: Return only O
        return O

    @staticmethod
    def backward(ctx, grad_O):
        # Copy the EXACT same backward logic from FlashAttentionTritonFunc
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        dO = grad_O

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