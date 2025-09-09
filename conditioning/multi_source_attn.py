import torch
from diffusers.models.attention_processor import AttnProcessor

class MultiSourceAttnProcessor(AttnProcessor):
    """
    Thay cho CrossAttn mặc định. Nó nhận 'encoder_hidden_states'
    là dict: {"text":C_txt, "garment":C_gar, "person":C_per,
              "t_embed":t_embed, "person_global":p_global}
    và trộn K,V từ từng nguồn bằng gating trước khi attention.
    """

    def __init__(self, gate_net):
        super().__init__()
        self.gate_net = gate_net

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        ctx = encoder_hidden_states
        C_txt = ctx["text"]  # [B,Lt,C]
        C_gar = ctx["garment"]  # [B,Lg,C]
        C_per = ctx["person"]  # [B,Lp,C]
        t_embed = ctx["t_embed"]  # [B,C]
        p_global = ctx["person_global"]  # [B,C]

        # ✅ FUSION: Tính adaptive weights
        alphas = self.gate_net(p_global, t_embed)  # [B,3]
        
        # Separate K,V projections cho từng source
        q = attn.to_q(hidden_states)
        k_txt = attn.to_k(C_txt)  # [B,Lt,head_dim*heads]
        k_gar = attn.to_k(C_gar)  # [B,Lg,head_dim*heads] 
        k_per = attn.to_k(C_per)  # [B,Lp,head_dim*heads]
        v_txt = attn.to_v(C_txt)
        v_gar = attn.to_v(C_gar)
        v_per = attn.to_v(C_per)

        # ✅ FUSION: Blend K,V theo alphas (như trong diagram)
        # Expand alphas for broadcasting
        a_t = alphas[:, 0:1, None]  # [B,1,1]
        a_g = alphas[:, 1:2, None]  # [B,1,1] 
        a_p = alphas[:, 2:3, None]  # [B,1,1]

        # Weighted average của K,V (adaptive fusion)
        k_fused = a_t * k_txt + a_g * k_gar + a_p * k_per
        v_fused = a_t * v_txt + a_g * v_gar + a_p * v_per

        # Standard attention với fused K,V
        q, k_fused, v_fused = (
            attn.head_to_batch_dim(q),
            attn.head_to_batch_dim(k_fused),
            attn.head_to_batch_dim(v_fused),
        )

        attn_scores = torch.baddbmm(
            torch.empty(
                q.shape[0], q.shape[1], k_fused.shape[1], 
                device=q.device, dtype=q.dtype
            ),
            q,
            k_fused.transpose(-1, -2),
            beta=0,
            alpha=attn.scale,
        )

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = attn_scores.softmax(dim=-1)
        hidden_states = torch.bmm(attn_probs, v_fused)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        return hidden_states
