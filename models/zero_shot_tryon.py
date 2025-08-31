import torch, torch.nn as nn
from diffusers import AutoencoderKL, DDPMScheduler
from modules.encoders.promt_encoder import TextEncoder
from modules.encoders.garment_encoder import GarmentEncoder
from modules.encoders.person_encoder import PersonEncoder

class ZeroShotTryOn(nn.Module):
    def __init__(self, unet, vae_name="runwayml/stable-diffusion-v1-5"):
        super().__init__()
        self.unet = unet
        self.vae = AutoencoderKL.from_pretrained(vae_name, subfolder="vae")
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear")
        self.txt = TextEncoder()
        self.gar = GarmentEncoder(out_dim=768)
        self.per = PersonEncoder(in_ch=3+18+20, out_dim=768)  # ví dụ

    def _make_context(self, person_tensor, garment_tensor, prompts, t_embed):
        # encode
        C_txt = self.txt.encode(prompts, device=person_tensor.device)        # [B,Lt,768]
        C_gar = self.gar(garment_tensor)                                     # [B,Lg,768]
        C_per = self.per(person_tensor)                                      # [B,Lp,768]
        
        # Concatenate thành 1 tensor thay vì dict
        context = torch.cat([C_txt, C_gar, C_per], dim=1)  # [B, Lt+Lg+Lp, 768]
        return context

    def forward(self, imgs, person_cond, garment_img, prompts):
        """
        imgs: ground-truth ảnh người mặc đồ target (để tính loss)
        person_cond: tensor điều kiện (pose+parsing+...): [B,C,H,W]
        garment_img: [B,3,H,W]
        prompts: list[str]
        """
        # VAE encode -> latent
        latents = self.vae.encode(imgs).latent_dist.sample() * 0.18215

        # sample noise & timestep
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.size(0),), device=latents.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # context (giờ là tensor thay vì dict)
        ctx = self._make_context(person_cond, garment_img, prompts, None)

        # UNet noise prediction (epsilon)
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=ctx).sample

        # simple L2
        loss = torch.nn.functional.mse_loss(model_pred, noise)
        return loss
