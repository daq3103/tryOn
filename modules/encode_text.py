import torch

@torch.no_grad()
def encode_text(prompts, pipe, device):
    tok = pipe.tokenizer(prompts, padding="max_length",
                         max_length=pipe.tokenizer.model_max_length, return_tensors="pt").to(device)
    return pipe.text_encoder(**tok)[0] 

              