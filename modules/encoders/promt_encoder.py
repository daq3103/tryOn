import torch
from transformers import CLIPTextModel, CLIPTokenizer


class TextEncoder(torch.nn.Module):
    def __init__(self, name="openai/clip-vit-large-patch14"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(name)
        self.model = CLIPTextModel.from_pretrained(name)
        # self.out_dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, prompts: list[str], device):
        tok = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        out = self.model(**tok).last_hidden_state  # [B, L_txt, C]
        return out


# # Giữ lại function cũ để tương thích ngược
# @torch.no_grad()
# def encode_text(prompts, pipe, device):
#     tok = pipe.tokenizer(
#         prompts,
#         padding="max_length",
#         max_length=pipe.tokenizer.model_max_length,
#         return_tensors="pt",
#     ).to(device)
#     return pipe.text_encoder(**tok)[0]
