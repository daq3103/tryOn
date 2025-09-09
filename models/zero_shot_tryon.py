import torch, torch.nn as nn
from diffusers import AutoencoderKL, DDPMScheduler
from modules.encoders.promt_encoder import TextEncoder
from modules.encoders.garment_encoder import GarmentEncoder
from modules.encoders.person_encoder import PersonEncoder
from conditioning.fuse import GatedFusion
from conditioning.multi_source_attn import MultiSourceAttnProcessor
from diffusers.models.attention_processor import AttnProcessor

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

        # Add time embedding projection to match 768 dim
        self.time_proj = nn.Linear(1280, 768)

        # Gate net with correct dimension
        self.gate_net = GatedFusion(c_dim=768)

        # Attach multi-source attention
        self._setup_attention_processors()

    def _setup_attention_processors(self):
        current_processors = self.unet.attn_processors
        attn_procs = {}
        cross_attn_count = 0
        
        for name in current_processors.keys():
            if "attn2" in name:  # cross-attention
                attn_procs[name] = MultiSourceAttnProcessor(gate_net=self.gate_net)
                cross_attn_count += 1
            else:  # self-attention
                attn_procs[name] = AttnProcessor()
        
        # Save total count BEFORE setting processors
        total_processors = len(attn_procs)
        
        # Set all processors
        self.unet.set_attn_processor(attn_procs)
        print(f"✅ Set {cross_attn_count} custom cross-attention processors out of {total_processors} total")

    
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
        return loss

    def _make_context(self, person_tensor, garment_tensor, prompts, timesteps):
        # encode
        C_txt = self.txt.encode(prompts, device=person_tensor.device)
        C_gar = self.gar(garment_tensor)
        C_per = self.per(person_tensor)

        # Get timestep embeddings
        t_embed = self.unet.time_proj(timesteps)
        t_embed = self.unet.time_embedding(t_embed)  # [B, 1280]
        
        # Project to 768 dim to match p_global
        t_embed = self.time_proj(t_embed)  # [B, 768]
        
        # Global person features
        p_global = C_per.mean(dim=1)  # [B, 768]
        context = {
            "text": C_txt,
            "garment": C_gar,
            "person": C_per,
            "t_embed": t_embed,
            "person_global": p_global,
        }
        return context
