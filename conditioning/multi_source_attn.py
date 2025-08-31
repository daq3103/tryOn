import torch, torch.nn as nn
from diffusers.models.attention_processor import AttnProcessor

class MultiSourceAttnProcessor(AttnProcessor):
    """
    Thay cho CrossAttn mặc định. Nó nhận 'encoder_hidden_states'
    là dict: {"text":C_txt, "garment":C_gar, "person":C_per,
              "t_embed":t_embed, "person_global":p_global}
    và trộn K,V từ từng nguồn bằng gating trước khi attention.
    """
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None):
        B, N, C = hidden_states.shape

        ctx = encoder_hidden_states
        C_txt = ctx["text"]        # [B,Lt,C]
        C_gar = ctx["garment"]     # [B,Lg,C]
        C_per = ctx["person"]      # [B,Lp,C]
        t_embed = ctx["t_embed"]   # [B,C]
        p_global = ctx["person_global"]  # [B,C]

        # proj q,k,v cho từng nguồn
        q = attn.to_q(hidden_states)
        k_txt = attn.to_k(C_txt); v_txt = attn.to_v(C_txt)
        k_gar = attn.to_k(C_gar); v_gar = attn.to_v(C_gar)
        k_per = attn.to_k(C_per); v_per = attn.to_v(C_per)

        # gating (dùng 1 mlp nhỏ gắn vào attn.processor_state)
        alphas = attn.processor_state["gate"](p_global, t_embed)  # [B,3]
        a_t, a_g, a_p = alphas[:,0:1,None,None], alphas[:,1:2,None,None], alphas[:,2:3,None,None]

        # căn chỉnh batch dimension
        def blend(k1, k2, k3, a1, a2, a3):
            # k*: [B,L,C]; a*: [B,1,1,1]
            return a1*k1 + a2*k2 + a3*k3

        k = blend(k_txt, k_gar, k_per, a_t, a_g, a_p)
        v = blend(v_txt, v_gar, v_per, a_t, a_g, a_p)

        # phần còn lại giống cross-attention tiêu chuẩn
        q, k, v = attn.head_to_batch_dim(q), attn.head_to_batch_dim(k), attn.head_to_batch_dim(v)
        attn_scores = torch.baddbmm(
            torch.empty(q.shape[0], q.shape[1], k.shape[1], device=q.device, dtype=q.dtype),
            q, k.transpose(-1, -2), beta=0, alpha=attn.scale
        )
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = attn_scores.softmax(dim=-1)
        hidden_states = torch.bmm(attn_probs, v)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
