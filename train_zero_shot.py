import torch
from torch.utils.data import DataLoader
from data.datasets import VTONDataset
from models.zero_shot_tryon import ZeroShotTryOn
from diffusers import UNet2DConditionModel
from conditioning.multi_source_attn import MultiSourceAttnProcessor
from conditioning.fuse import GatedFusion
import os
from PIL import Image

def create_dummy_data_if_needed(data_path):
    """Tạo dummy data nếu không tồn tại"""
    pairs_file = os.path.join(data_path, "train_pairs.txt")
    
    if not os.path.exists(pairs_file):
        print(f"⚠️ train_pairs.txt not found. Creating dummy data in {data_path}")
        
        # Tạo thư mục nếu chưa có
        os.makedirs(data_path, exist_ok=True)
        for folder in ["person", "garment", "target", "pose", "parsing"]:
            os.makedirs(os.path.join(data_path, folder), exist_ok=True)
        
        # Tạo file pairs
        with open(pairs_file, 'w') as f:
            for i in range(20):  # 20 samples để test
                f.write(f"person{i:03d}.jpg\tgarment{i:03d}.jpg\t\"a person wearing fashionable clothes\"\n")
        
        # Tạo ảnh dummy
        for i in range(20):
            # Tạo ảnh với màu khác nhau
            color = (i*10 % 255, (i*20) % 255, (i*30) % 255)
            img = Image.new('RGB', (512, 512), color=color)
            
            img.save(os.path.join(data_path, "person", f"person{i:03d}.jpg"))
            img.save(os.path.join(data_path, "garment", f"garment{i:03d}.jpg"))
            img.save(os.path.join(data_path, "target", f"person{i:03d}_garment{i:03d}.jpg"))
        
        print(f"✅ Created dummy data: {pairs_file}")
    
    return pairs_file

def setup_model_and_processor(use_small_unet=False):
    """Khởi tạo UNet và gắn MultiSourceAttnProcessor"""
    try:
        if use_small_unet:
            unet = UNet2DConditionModel(
                sample_size=64,
                in_channels=4,
                out_channels=4,
                layers_per_block=1,
                block_out_channels=(128, 256),
                down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
                cross_attention_dim=768,
            )
            print("Using small UNet for faster training")
        else:
            unet = UNet2DConditionModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                subfolder="unet"
            )
            print("Using full-size UNet")
        
        # Khởi tạo gating network
        gate_net = GatedFusion(c_dim=unet.config.cross_attention_dim)
        
        # Gắn processor với gate network
        processor_count = 0
        for name, module in unet.named_modules():
            if "attn2" in name and hasattr(module, 'set_processor'):
                try:
                    # Tạo processor với gate network
                    processor = MultiSourceAttnProcessor(gate_net=gate_net)
                    module.set_processor(processor)
                    processor_count += 1
                except Exception as e:
                    print(f"Warning: Could not set processor for {name}: {e}")
        
        print(f"✅ Set {processor_count} custom attention processors")
        return unet, gate_net
        
    except Exception as e:
        print(f"❌ Model setup failed: {e}")
        print("💡 Falling back to simple setup without custom processors")
        
        # Fallback: UNet đơn giản không có custom processor
        unet = UNet2DConditionModel(
            sample_size=32 if use_small_unet else 64,
            in_channels=4,
            out_channels=4,
            layers_per_block=1,
            block_out_channels=(64, 128) if use_small_unet else (128, 256),
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=768,
        )
        gate_net = GatedFusion(c_dim=768)
        return unet, gate_net

def get_trainable_params(unet, gate_net, model):
    """Lấy parameters cần train (freeze UNet, chỉ train gate + encoders)"""
    # Freeze UNet parameters
    for param in unet.parameters():
        param.requires_grad = False
    
    # Train gate network và encoders
    trainable_params = list(gate_net.parameters())
    trainable_params.extend(list(model.txt.parameters()))
    trainable_params.extend(list(model.gar.parameters()))
    trainable_params.extend(list(model.per.parameters()))
    
    return trainable_params

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        try:
            imgs = batch["target_img"].to(device)
            loss = model(
                imgs=imgs,
                person_cond=batch["person_cond"].to(device),
                garment_img=batch["garment_img"].to(device),
                prompts=batch["prompt"]
            )
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping để tránh exploding gradients
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 
                max_norm=1.0
            )
            
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    return total_loss / max(num_batches, 1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dummy_data", help="Path to data directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--small_unet", action="store_true", help="Use small UNet for faster training")
    parser.add_argument("--image_size", type=int, default=256, help="Image size (use 256 for faster training)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Tạo dummy data nếu cần
    pairs_file = create_dummy_data_if_needed(args.data_path)
    
    # Chuẩn bị dataloader - FIX: sử dụng đường dẫn đầy đủ cho pairs_txt
    try:
        ds = VTONDataset(root=args.data_path, pairs_txt=pairs_file, size=args.image_size)  # Thay đổi ở đây
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        print(f"Dataset size: {len(ds)} samples")
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        print(f"Expected pairs file: {pairs_file}")
        print(f"File exists: {os.path.exists(pairs_file)}")
        exit(1)

    # Khởi tạo model và processor
    unet, gate_net = setup_model_and_processor(use_small_unet=args.small_unet)
    model = ZeroShotTryOn(unet)
    model.to(device)
    
    # Optimizer cho parameters cần train
    trainable_params = get_trainable_params(unet, gate_net, model)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Training loop
    num_epochs = args.epochs
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("=" * 50)
        
        # Training
        train_loss = train_epoch(model, dl, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Validation (mỗi 5 epochs)
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in dl:
                    try:
                        imgs = batch["target_img"].to(device)
                        loss = model(
                            imgs=imgs,
                            person_cond=batch["person_cond"].to(device),
                            garment_img=batch["garment_img"].to(device),
                            prompts=batch["prompt"]
                        )
                        val_loss += loss.item()
                        val_batches += 1
                        
                        if val_batches >= 3:  # Chỉ validate 3 batches cho nhanh
                            break
                            
                    except Exception as e:
                        print(f"Validation error: {e}")
                        continue
            
            val_loss = val_loss / max(val_batches, 1)
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, 'best_model.pth')
                print("✅ Saved best model!")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')
            print(f"💾 Saved checkpoint at epoch {epoch+1}")

    print("🎉 Training completed!")