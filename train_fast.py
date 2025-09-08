#!/usr/bin/env python3
"""
Script training nhanh với UNet nhỏ để test concept
"""

import torch
from torch.utils.data import DataLoader
from data.datasets import VTONDataset
from models.zero_shot_tryon import ZeroShotTryOn
from diffusers import UNet2DConditionModel
from conditioning.multi_source_attn import MultiSourceAttnProcessor
from conditioning.fuse import GatedFusion
import tempfile
import os
from PIL import Image


def create_dummy_data():
    """Tạo dữ liệu giả để test training"""
    # Tạo thư mục temp_data trong project
    temp_dir = "temp_data"
    os.makedirs(temp_dir, exist_ok=True)

    # Tạo thư mục con
    for folder in ["person", "garment", "target", "pose", "parsing"]:
        os.makedirs(os.path.join(temp_dir, folder), exist_ok=True)

    # Tạo file pairs
    pairs_file = os.path.join(temp_dir, "train_pairs.txt")
    with open(pairs_file, "w") as f:
        for i in range(10):  # 10 samples
            f.write(f'person{i}.jpg\tgarment{i}.jpg\t"a person wearing clothes"\n')

    # Tạo ảnh test (64x64 để nhanh)
    for i in range(10):
        img = Image.new("RGB", (64, 64), color=(i * 25, i * 25, i * 25))
        img.save(os.path.join(temp_dir, "person", f"person{i}.jpg"))
        img.save(os.path.join(temp_dir, "garment", f"garment{i}.jpg"))
        img.save(os.path.join(temp_dir, "target", f"person{i}_garment{i}.jpg"))

    return temp_dir, pairs_file


def setup_small_model():
    """Setup model nhỏ cho training nhanh"""
    # UNet rất nhỏ - có thể giảm thêm cho GPU yếu
    unet = UNet2DConditionModel(
        sample_size=32,  # Nhỏ hơn nữa
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(64, 128),  # Ít channels hơn
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=768,
    )

    # Cho GPU < 2GB, uncomment dòng dưới:
    # block_out_channels=(32, 64)  # Còn nhỏ hơn nữa
    # cross_attention_dim=384      # Giảm attention dim

    # Gating network
    gate_net = GatedFusion(c_dim=unet.config.cross_attention_dim)

    # Skip custom processor cho bây giờ - dùng standard cross attention
    print("Using standard cross-attention (skipping custom processor for simplicity)")

    return unet, gate_net


def train_fast():
    """Training nhanh với setup tối ưu"""
    print("🚀 Starting fast training...")

    # Tạo dữ liệu giả
    data_dir, pairs_file = create_dummy_data()
    print(f"Created dummy data in: {data_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset nhỏ
    ds = VTONDataset(root=data_dir, pairs_txt=pairs_file, size=64)
    dl = DataLoader(
        ds, batch_size=4, shuffle=True, num_workers=0
    )  # num_workers=0 để tránh lỗi
    print(f"Dataset: {len(ds)} samples")

    # Model nhỏ
    unet, gate_net = setup_small_model()
    model = ZeroShotTryOn(unet)
    model.to(device)

    # Freeze UNet, train encoders + gate
    for param in unet.parameters():
        param.requires_grad = False

    trainable_params = list(gate_net.parameters())
    trainable_params.extend(list(model.txt.parameters()))
    trainable_params.extend(list(model.gar.parameters()))
    trainable_params.extend(list(model.per.parameters()))

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)  # LR cao hơn
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    # Training loop ngắn
    num_epochs = 10
    print(f"Training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dl):
            try:
                imgs = batch["target_img"].to(device)
                loss = model(
                    imgs=imgs,
                    person_cond=batch["person_cond"].to(device),
                    garment_img=batch["garment_img"].to(device),
                    prompts=batch["prompt"],
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if batch_idx % 2 == 0:
                    print(
                        f"  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                    )

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss": avg_loss,
                },
                f"fast_checkpoint_epoch_{epoch+1}.pth",
            )
            print(f"Saved checkpoint at epoch {epoch+1}")

    print("✅ Fast training completed!")
    return model


if __name__ == "__main__":
    train_fast()
