import torch, torch.nn as nn

class GatedFusion(nn.Module):
    """
    Tính trọng số alpha_text, alpha_garment, alpha_person (softmax) từ
    global pooled person features + timestep embedding (hoặc sigma).
    """
    def __init__(self, c_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_dim*2, c_dim), nn.SiLU(),
            nn.Linear(c_dim, 3)
        )

    def forward(self, person_global, t_embed):
        g = torch.cat([person_global, t_embed], dim=-1)
        logits = self.fc(g)            # [B,3]
        alphas = torch.softmax(logits, dim=-1)
        return alphas                  # [B,3]
