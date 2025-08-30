import os, torch
from torch import nn
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler
from modules.attn import ZeroGarmentAttn
from dataset import TryOnDataset
from modules.encode_text import encode_text
from modules.encoder import GarmentEncoder

from modules.vae import encode_vae   


device  = "cuda" if torch.cuda.is_available() else "cpu"
dtype   = torch.float16 if device == "cuda" else torch.float32
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
pipe.unet.requires_grad_(False)   # freeze UNet gốc

# --- gắn Zero-Garment Attn vào các cross-attn layer (attn2) ---
garment_dim = 512  
processors = {}
for name, module in pipe.unet.named_modules():
    if hasattr(module, "set_processor"):
        if "attn2" in name:
            proc = ZeroGarmentAttn(
                hidden_size=module.to_q.in_features, 
                ctx_size=garment_dim
            )
            module.set_processor(proc)
            processors[name] = proc 

# chỉ train các processor (và encoder áo nếu có)
params = [p for m in processors.values() for p in m.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-2)

noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

ds = TryOnDataset("data/train", size=512)
dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=4, drop_last=True)




garment_encoder = GarmentEncoder().to(device)
params += list(garment_encoder.parameters())         

scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

for epoch in range(1):
    for step, batch in enumerate(dl):
        person = batch["person"].to(device)
        cloth  = batch["cloth"].to(device)
        prompts = batch["prompt"]

        with torch.no_grad():
            z = encode_vae(pipe.vae, person, device=device)  # [B,4,64,64]
            text_ctx = encode_text(list(prompts), pipe, device=device)

        garment_ctx = garment_encoder(cloth)           # [B,S,Dg]

        bsz = z.shape[0]
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
        noise = torch.randn_like(z)
        z_noisy = noise_scheduler.add_noise(z, noise, t)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
            out = pipe.unet(
                sample=z_noisy, timestep=t, encoder_hidden_states=text_ctx,
                cross_attention_kwargs={"garment_ctx": garment_ctx},
                return_dict=True
            )
            noise_pred = out.sample
            loss = nn.functional.mse_loss(noise_pred.float(), noise.float())

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0:
            print(f"step {step} | loss {loss.item():.4f}")

# Lưu nhẹ các trọng số processor + garment encoder
os.makedirs("checkpoints", exist_ok=True)
state = {
    "processors": {k: v.state_dict() for k, v in processors.items()},
    "garment_encoder": garment_encoder.state_dict(),
}
torch.save(state, "checkpoints/zero_xattn_tryon.pt")
print("Saved to checkpoints/zero_xattn_tryon.pt")
