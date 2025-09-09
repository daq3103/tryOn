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
        C_txt = ctx["text"]  # [B,77,768]
        C_gar = ctx["garment"]  # [B,64,768]
        C_per = ctx["person"]  # [B,4096,768]
        t_embed = ctx["t_embed"]  # [B,768]
        p_global = ctx["person_global"]  # [B,768]

        # ✅ FUSION: Tính adaptive weights
        alphas = self.gate_net(p_global, t_embed)  # [B,3]
        
        # Concatenate contexts
        combined_context = torch.cat([C_txt, C_gar, C_per], dim=1)

        # Standard projections
        q = attn.to_q(hidden_states)
        k = attn.to_k(combined_context)
        v = attn.to_v(combined_context)

        q, k, v = (
            attn.head_to_batch_dim(q),
            attn.head_to_batch_dim(k),
            attn.head_to_batch_dim(v),
        )

        attn_scores = torch.baddbmm(
            torch.empty(q.shape[0], q.shape[1], k.shape[1], device=q.device, dtype=q.dtype),
            q, k.transpose(-1, -2), beta=0, alpha=attn.scale,
        )

        # ✅ APPLY GATING: Modulate attention scores by regions
        B = len(alphas)
        Lt, Lg, Lp = C_txt.shape[1], C_gar.shape[1], C_per.shape[1]
        
        # Split by source regions
        attn_txt = attn_scores[:, :, :Lt]                    # [B*heads, seq, 77]
        attn_gar = attn_scores[:, :, Lt:Lt+Lg]              # [B*heads, seq, 64]  
        attn_per = attn_scores[:, :, Lt+Lg:Lt+Lg+Lp]       # [B*heads, seq, 4096]

        # Expand alphas for multi-head
        heads = attn_scores.shape[0] // B
        alphas_expanded = alphas.repeat_interleave(heads, dim=0)  # [B*heads, 3]
        
        # ✅ GATED FUSION: Apply adaptive weights
        attn_txt = attn_txt * alphas_expanded[:, 0:1, None]  # α_text * attention_to_text
        attn_gar = attn_gar * alphas_expanded[:, 1:2, None]  # α_garment * attention_to_garment
        attn_per = attn_per * alphas_expanded[:, 2:3, None]  # α_person * attention_to_person
        
        # Recombine
        attn_scores = torch.cat([attn_txt, attn_gar, attn_per], dim=-1)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = attn_scores.softmax(dim=-1)
        hidden_states = torch.bmm(attn_probs, v)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        return hidden_states
