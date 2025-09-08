import torch.nn as nn
import torchvision.models as tvm


class GarmentEncoder(nn.Module):
    """
    Biến ảnh áo/quần (H,W,3) -> chuỗi token [B, L_g, C]
    Cách đơn giản: backbone CNN + flatten không gian.
    """

    def __init__(self, out_dim=768):
        super().__init__()
        backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        self.cnn = nn.Sequential(*(list(backbone.children())[:-2]))  # [B,2048,h,w]
        self.proj = nn.Conv2d(2048, out_dim, 1)
        self.out_dim = out_dim

    def forward(self, x):  # x: [B,3,H,W]
        f = self.cnn(x)  # [B,2048,h,w]
        f = self.proj(f)  # [B,C,h,w]
        tokens = f.flatten(2).transpose(1, 2)  # [B, L=h*w, C]
        return tokens
