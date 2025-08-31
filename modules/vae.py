import torch


@torch.no_grad()
def encode_vae(vae, x, device):
    """Encode an image tensor ``x`` using a VAE.

    Parameters
    ----------
    vae : Any
        The VAE model used for encoding.
    x : torch.Tensor
        Image tensor to encode.
    device : torch.device | str
        Device on which to run the encoding.

    Returns
    -------
    torch.Tensor
        The latent representation scaled by ``0.18215``.
    """
    z = vae.encode(x.to(device, dtype=vae.dtype)).latent_dist.sample()
    return z * 0.18215
