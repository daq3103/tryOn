from diffusers import UNet2DConditionModel

from conditioning.fuse import GatedFusion

# Thử với model public khác
unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="unet"  # Model public
)
gate_net = GatedFusion(c_dim=unet.config.cross_attention_dim)
print("gate_net:", gate_net)
