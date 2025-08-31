# inference.py
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"
controlnet = ControlNetModel.from_pretrained(
    "checkpoints/controlnet_tryon_baseline",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)


def load_pil(path, size=512):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return img


person = load_pil("data/val/person.png")
cloth = load_pil("data/val/cloth.png")

out = pipe(
    prompt="a person wearing the given garment",
    image=person,  # init image (img2img)
    control_image=cloth,  # control = garment image
    num_inference_steps=30,
    guidance_scale=5.0,  # CFG
    controlnet_conditioning_scale=1.0,  # strength của garment guidance
    strength=0.8,  # mức giữ lại cấu trúc người từ ảnh gốc
)
out.images[0].save("tryon_baseline.png")
print("Saved tryon_baseline.png")
