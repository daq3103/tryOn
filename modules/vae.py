
import torch

@torch.no_grad()
def encode_vae(x, pipe, device):
    z = pipe.vae.encode(x.to(device, dtype=pipe.vae.dtype)).latent_dist.sample()
    return z * 0.18215   