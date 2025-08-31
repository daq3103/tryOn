#!/usr/bin/env python3
"""
Training script tối ưu cho Kaggle
- Sử dụng GPU hiệu quả
- Memory management
- Progress tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import gc
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import project modules
from data.datasets import VTONDataset
from models.zero_shot_tryon import ZeroShotTryOn
from diffusers import UNet2DConditionModel
from conditioning.multi_source_attn import MultiSourceAttnProcessor
from conditioning.fuse import GatedFusion
import tempfile
from PIL import Image

def setup_kaggle_gpu():
    """Setup GPU cho Kaggle"""
    print("🔧 Setting up Kaggle GPU...")
    
    # Check GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU available: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Optimize memory
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU available, using CPU")
    
    return device

def create_kaggle_dataset(num_samples=50):
    """Tạo dataset lớn hơn cho Kaggle"""
    print("📊 Creating Kaggle dataset...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Tạo thư mục
        for folder in ["person", "garment", "target", "pose", "parsing"]:
            os.makedirs(os.path.join(temp_dir, folder), exist_ok=True)
        
        # Tạo file pairs
        pairs_file = os.path.join(temp_dir, "train_pairs.txt")
        with open(pairs_file, 'w') as f:
            for i in range(num_samples):
                f.write(f"person{i}.jpg\tgarment{i}.jpg\t\"a person wearing clothes\"\n")
        
        # Tạo ảnh test với kích thước lớn hơn
        for i in range(num_samples):
            # Tạo ảnh với màu sắc đa dạng
            color = (i * 5 % 255, (i * 7) % 255, (i * 11) % 255)
            img = Image.new('RGB', (128, 128), color=color)
            img.save(os.path.join(temp_dir, "person", f"person{i}.jpg"))
            img.save(os.path.join(temp_dir, "garment", f"garment{i}.jpg"))
            img.save(os.path.join(temp_dir, "target", f"person{i}_garment{i}.jpg"))
        
        return temp_dir, pairs_file

def setup_kaggle_model(device):
    """Setup model tối ưu cho Kaggle"""
    print("🧠 Setting up Kaggle model...")
    
    # UNet vừa phải cho Kaggle
    unet = UNet2DConditionModel(
        sample_size=64,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,  # Tăng depth một chút
        block_out_channels=(128, 256, 512),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=768,
    )
    
    # Gating network
    gate_net = GatedFusion(c_dim=unet.config.cross_attention_dim)
    
    # Gắn processor
    for name, module in unet.named_modules():
        if "attn2" in name:
            processor = MultiSourceAttnProcessor()
            processor.processor_state = {"gate": gate_net}
            module.set_processor(processor)
    
    # Model
    model = ZeroShotTryOn(unet)
    model.to(device)
    
    # Freeze UNet
    for param in unet.parameters():
        param.requires_grad = False
    
    return model, gate_net

def train_kaggle_epoch(model, loader, optimizer, device, epoch, num_epochs):
    """Training epoch với progress bar"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Progress bar
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # Move to device
            imgs = batch["target_img"].to(device)
            person_cond = batch["person_cond"].to(device)
            garment_img = batch["garment_img"].to(device)
            prompts = batch["prompt"]
            
            # Forward pass
            loss = model(imgs, person_cond, garment_img, prompts)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{total_loss/num_batches:.4f}'
            })
            
            # Memory cleanup
            if batch_idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    return total_loss / max(num_batches, 1)

def plot_training_progress(losses, save_path="training_progress.png"):
    """Plot training progress"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('Training Loss Progress', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def train_on_kaggle():
    """Main training function cho Kaggle"""
    print("🚀 Starting Kaggle training...")
    
    # Setup
    device = setup_kaggle_gpu()
    data_dir, pairs_file = create_kaggle_dataset(num_samples=100)
    
    # Dataset
    ds = VTONDataset(root=data_dir, pairs_txt="train_pairs.txt", size=128)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2)
    print(f"Dataset: {len(ds)} samples")
    
    # Model
    model, gate_net = setup_kaggle_model(device)
    
    # Optimizer
    trainable_params = list(gate_net.parameters())
    trainable_params.extend(list(model.txt.parameters()))
    trainable_params.extend(list(model.gar.parameters()))
    trainable_params.extend(list(model.per.parameters()))
    
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
    
    # Training
    num_epochs = 30
    losses = []
    best_loss = float('inf')
    
    print(f"\n🎯 Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Training
        avg_loss = train_kaggle_epoch(model, dl, optimizer, device, epoch, num_epochs)
        losses.append(avg_loss)
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f"Epoch {epoch+1} completed - Loss: {avg_loss:.4f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
            }, 'kaggle_best_model.pth')
            print("💾 Saved best model!")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'losses': losses,
            }, f'kaggle_checkpoint_epoch_{epoch+1}.pth')
            print(f"💾 Saved checkpoint at epoch {epoch+1}")
        
        # Plot progress
        if (epoch + 1) % 5 == 0:
            plot_training_progress(losses)
    
    print("\n🎉 Kaggle training completed!")
    print(f"Best loss: {best_loss:.4f}")
    
    # Final plot
    plot_training_progress(losses, "final_training_progress.png")
    
    return model, losses

if __name__ == "__main__":
    model, losses = train_on_kaggle()
