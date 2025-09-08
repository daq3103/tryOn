import torch, torch.nn as nn
from diffusers import AutoencoderKL, DDPMScheduler
from modules.encoders.promt_encoder import TextEncoder
from modules.encoders.garment_encoder import GarmentEncoder
from modules.encoders.person_encoder import PersonEncoder
from conditioning.fuse import GatedFusion
from conditioning.multi_source_attn import MultiSourceAttnProcessor


class ZeroShotTryOn(nn.Module):
    def __init__(self, unet, vae_name="runwayml/stable-diffusion-v1-5"):
        super().__init__()
        self.unet = unet
        self.vae = AutoencoderKL.from_pretrained(vae_name, subfolder="vae")
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000, beta_schedule="scaled_linear"
        )

        self.txt = TextEncoder()
        self.gar = GarmentEncoder(out_dim=768)
        self.per = PersonEncoder(in_ch=3 + 18 + 20, out_dim=768)

        # Only ONE gate_net living inside the model
        self.gate_net = GatedFusion(c_dim=self.unet.config.cross_attention_dim)

        # Attach multi-source attention ONLY to cross-attn ("attn2") processors
        self._setup_attention_processors()

    def _setup_attention_processors(self):
        attn_procs = {}
        # unet.attn_processors exposes the exact keys diffusers expects
        for name in self.unet.attn_processors.keys():
            if "attn2" in name:  # cross-attention only
                attn_procs[name] = MultiSourceAttnProcessor(gate_net=self.gate_net)
            else:
                attn_procs[name] = None  # keep default for others
        self.unet.set_attn_processor(attn_procs)
        num_custom = sum(p is not None for p in attn_procs.values())
        print(f"✅ Set {num_custom} custom cross-attention processors")

    def _make_context(self, person_tensor, garment_tensor, prompts, timesteps):
        # encode
        C_txt = self.txt.encode(prompts, device=person_tensor.device)  # [B,Lt,768]
        C_gar = self.gar(garment_tensor)  # [B,Lg,768]
        C_per = self.per(person_tensor)  # [B,Lp,768]

        # Get timestep embeddings
        t_embed = self.unet.time_proj(timesteps)
        t_embed = self.unet.time_embedding(t_embed)  # [B,768]

        # Global person features
        p_global = C_per.mean(dim=1)  # [B,768]

        # RETURN AS DICT FOR FUSION
        context = {
            "text": C_txt,
            "garment": C_gar,
            "person": C_per,
            "t_embed": t_embed,
            "person_global": p_global,
        }
        return context

    def forward(self, imgs, person_cond, garment_img, prompts):
        latents = self.vae.encode(imgs).latent_dist.sample() * 0.18215
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (latents.size(0),),
            device=latents.device,
        )
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # NOW WITH TIMESTEPS FOR FUSION
        ctx = self._make_context(person_cond, garment_img, prompts, timesteps)

        model_pred = self.unet(
            noisy_latents, timesteps, encoder_hidden_states=ctx
        ).sample

        loss = torch.nn.functional.mse_loss(model_pred, noise)
        print("loss noising:", loss.item())
        return loss
