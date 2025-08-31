import torch.nn as nn

class PersonEncoder(nn.Module):
    """
    Nhập các kênh điều kiện người: pose map (K kênh), human parsing (N lớp),
    hoặc ảnh người đã che mặt áo gốc. Ghép lại thành tensor [B,C,H,W].
    """
    def __init__(self, in_ch=3+18+20, out_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(512, out_dim, 1)
        )
        self.out_dim = out_dim

    def forward(self, x):  # x: [B,in_ch,H,W]
        f = self.net(x)                          # [B,C,h,w]
        tokens = f.flatten(2).transpose(1,2)     # [B,L_p,C]
        return tokens
