import torch
from torch.utils.data import DataLoader
from data.datasets import VTONDataset
from models.zero_shot_tryon import ZeroShotTryOn
import os
from diffusers import UNet2DConditionModel

def get_trainable_params(model):
    # Freeze toàn bộ UNet & VAE
    for p in model.unet.parameters():
        p.requires_grad = False
    for p in model.vae.parameters():
        p.requires_grad = False

    # Train gate_net + encoders (txt, gar, per)
    trainable = list(model.gate_net.parameters())
    trainable += list(model.txt.parameters())
    trainable += list(model.gar.parameters())
    trainable += list(model.per.parameters())
    return trainable


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
                prompts=batch["prompt"],
            )

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping để tránh exploding gradients
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
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
    parser.add_argument(
        "--data_path",
        type=str,
        default="./converted_train_data/",
        help="Path to data directory",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--small_unet", action="store_true", help="Use small UNet for faster training"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size (use 256 for faster training)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Check data path exists
    if not os.path.exists(args.data_path):
        print(f"❌ Data path does not exist: {args.data_path}")
        exit(1)

    pairs_file = os.path.join(args.data_path, "train_pairs.txt")
    if not os.path.exists(pairs_file):
        print(f"❌ Pairs file not found: {pairs_file}")
        exit(1)

    # Initialize dataset
    try:
        ds = VTONDataset(
            root=args.data_path, pairs_txt=pairs_file, size=args.image_size
        )
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        print(f"Dataset size: {len(ds)} samples")
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        exit(1)

    # Khởi tạo model và processor
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet"
    )
    model = ZeroShotTryOn(unet).to(device)

    # lấy parameters cần train
    trainable_params = get_trainable_params(model)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Training loop
    num_epochs = args.epochs
    best_loss = float("inf")

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
                        imgs = batch["target_img"].to(device)  # ảnh mục tiêu
                        loss = model(
                            imgs=imgs,
                            person_cond=batch["person_cond"].to(device), 
                            garment_img=batch["garment_img"].to(device),
                            prompts=batch["prompt"],
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
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    "best_model.pth",
                )
                print("✅ Saved best model!")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                },
                f"checkpoint_epoch_{epoch+1}.pth",
            )
            print(f"💾 Saved checkpoint at epoch {epoch+1}")

    print("🎉 Training completed!")
