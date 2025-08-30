import torch
import torch.nn as nn
from diffusers.models.attention_processor import AttnProcessor

class ZeroGarmentAttn(AttnProcessor):
    def __init__(self, hidden_size: int, ctx_size: int):
        super().__init__()
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(ctx_size,    hidden_size, bias=False)
        self.to_v = nn.Linear(ctx_size,    hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.proj.weight)   # zero-init (trọng số đầu ra)

    def __call__(
        self,
        attn,
        hidden_states,                       # [B, N, D_hid] (từ UNet)
        encoder_hidden_states=None,          # text ctx gốc
        attention_mask=None,
        temb=None,
        **cross_attention_kwargs             # nhận garment_ctx ở đây
    ):
        # 1) kết quả cross/self-attn gốc (text)
        out = attn.processor(
            attn, hidden_states, encoder_hidden_states,
            attention_mask=attention_mask, temb=temb, **cross_attention_kwargs
        )

        # 2) cross-attn bổ sung với garment
        garment_ctx = cross_attention_kwargs.get("garment_ctx", None)
        if garment_ctx is None:
            return out

        # đảm bảo 3D: [B, S, D_ctx]
        if garment_ctx.dim() == 2:
            garment_ctx = garment_ctx.unsqueeze(1)

        B, N, D = hidden_states.shape
        Q = self.to_q(hidden_states)                   # [B,N,D]
        K = self.to_k(garment_ctx)                     # [B,S,D]
        V = self.to_v(garment_ctx)                     # [B,S,D]
        att = (Q @ K.transpose(1, 2)) / (D ** 0.5)     # [B,N,S]
        W = att.softmax(-1)
        G = W @ V                                      # [B,N,D]

        return out + self.proj(G)                      # residual (zero-init ⇒ an toàn)
