import torch, torch.nn as nn

class Refiner(nn.Module):
    """
    Nhận ảnh coarse (từ diffusion decode) + garment + person cond
    -> xuất ảnh tinh chỉnh. Dùng UNet nhẹ hoặc ResNet U-Net.
    """
    def __init__(self, in_ch=3+3+3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3,1,1), nn.ReLU(),
            nn.Conv2d(base, base, 3,1,1), nn.ReLU(),
            nn.Conv2d(base, 3, 1)
        )

    def forward(self, coarse, garment, person_rgb):
        x = torch.cat([coarse, garment, person_rgb], dim=1)
        return self.net(x)
