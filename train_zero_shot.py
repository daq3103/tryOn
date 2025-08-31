import torch
from torch.utils.data import DataLoader
from data.datasets import VTONDataset
from models.zero_shot_tryon import ZeroShotTryOn
from diffusers import UNet2DConditionModel
from conditioning.multi_source_attn import MultiSourceAttnProcessor
from conditioning.fuse import GatedFusion

def setup_model_and_processor(use_small_unet=False):
    """Khởi tạo UNet và gắn MultiSourceAttnProcessor"""
    if use_small_unet:
        # UNet nhỏ cho training nhanh
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
        # UNet full-size
        unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="unet"
        )
        print("Using full-size UNet")
    
    # Khởi tạo gating network
    gate_net = GatedFusion(c_dim=unet.config.cross_attention_dim)
    
    # Gắn processor vào tất cả cross-attention layers
    for name, module in unet.named_modules():
        if "attn2" in name:  # cross-attention layers
            processor = MultiSourceAttnProcessor()
            processor.processor_state = {"gate": gate_net}
            module.set_processor(processor)
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
    parser.add_argument("--data_path", type=str, default="/path/to/data", help="Path to data directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--small_unet", action="store_true", help="Use small UNet for faster training")
    parser.add_argument("--image_size", type=int, default=512, help="Image size")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Chuẩn bị dataloader
    ds = VTONDataset(root=args.data_path, pairs_txt="train_pairs.txt", size=args.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"Dataset size: {len(ds)} samples")

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
                for batch in dl:  # Sử dụng cùng dataloader cho demo
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
                        
                        if val_batches >= 5:  # Chỉ validate 5 batches
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
                print("Saved best model!")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')
