import torch, torch.nn as nn

class VTONLoss(nn.Module):
    def __init__(self, w_l2=1.0, w_lpips=0.5, w_perc=0.5):
        super().__init__()
        self.w_l2 = w_l2
        # LPIPS/perceptual có thể dùng torchvision/vgg trước huấn luyện
        self.l2 = nn.L1Loss()

    def forward(self, pred, gt):
        return self.w_l2 * self.l2(pred, gt)
