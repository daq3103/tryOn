class GarmentEncoder(nn.Module):
    def __init__(self, dim=garment_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((16,16)),
            nn.Conv2d(3, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, dim, 1)        # [B,dim,16,16] → tokens 256
        )
    def forward(self, x):
        f = self.net(x)                  # [B, Dg, 16, 16]
        B, Dg, H, W = f.shape
        return f.permute(0,2,3,1).reshape(B, H*W, Dg)  