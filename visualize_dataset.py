import matplotlib.pyplot as plt
import torch
from data.datasets import VTONDataset
import numpy as np


def visualize_dataset_samples(data_path, num_samples=10):
    """Hiển thị 10 ảnh đầu từ dataset"""

    # Load dataset
    pairs_file = f"{data_path}/train_pairs.txt"
    dataset = VTONDataset(root=data_path, pairs_txt=pairs_file, size=512)

    print(f"Dataset size: {len(dataset)} samples")

    # Tạo figure để hiển thị
    fig, axes = plt.subplots(4, num_samples, figsize=(20, 8))
    fig.suptitle("Dataset Samples Preview", fontsize=16)

    for i in range(min(num_samples, len(dataset))):
        try:
            # Load sample
            sample = dataset[i]

            # Convert tensor về numpy và chuyển về [H,W,C]
            person_rgb = sample["person_rgb"].permute(1, 2, 0).numpy()
            garment_img = sample["garment_img"].permute(1, 2, 0).numpy()
            target_img = sample["target_img"].permute(1, 2, 0).numpy()

            # Pose visualization (sum all channels)
            person_cond = sample["person_cond"]
            pose_channels = person_cond[3:21]  # 18 pose channels
            pose_vis = torch.sum(pose_channels, dim=0).numpy()
            pose_vis = np.clip(pose_vis, 0, 1)

            # Plot images
            axes[0, i].imshow(person_rgb)
            axes[0, i].set_title(f"Person {i+1}", fontsize=10)
            axes[0, i].axis("off")

            axes[1, i].imshow(garment_img)
            axes[1, i].set_title(f"Garment {i+1}", fontsize=10)
            axes[1, i].axis("off")

            axes[2, i].imshow(target_img)
            axes[2, i].set_title(f"Target {i+1}", fontsize=10)
            axes[2, i].axis("off")

            axes[3, i].imshow(pose_vis, cmap="hot")
            axes[3, i].set_title(f"Pose {i+1}", fontsize=10)
            axes[3, i].axis("off")

            # Print sample info
            prompt = sample["prompt"]
            print(f"Sample {i+1}: {prompt}")

        except Exception as e:
            print(f"Error loading sample {i+1}: {e}")
            # Fill with empty plots
            for row in range(4):
                axes[row, i].text(
                    0.5,
                    0.5,
                    f"Error\nSample {i+1}",
                    ha="center",
                    va="center",
                    transform=axes[row, i].transAxes,
                )
                axes[row, i].axis("off")

    # Add row labels
    row_labels = ["Person RGB", "Garment", "Target", "Pose Heatmap"]
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, fontsize=12, rotation=90, labelpad=20)

    plt.tight_layout()
    plt.show()

    # Print dataset statistics
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    print(f"Total samples: {len(dataset)}")
    print(f"Image size: {dataset.size}x{dataset.size}")

    # Check data directories
    import os

    dirs_to_check = ["person", "garment", "target", "pose", "parsing"]
    for dir_name in dirs_to_check:
        dir_path = os.path.join(data_path, dir_name)
        if os.path.exists(dir_path):
            file_count = len(
                [
                    f
                    for f in os.listdir(dir_path)
                    if f.endswith((".jpg", ".png", ".json"))
                ]
            )
            print(f"{dir_name}: {file_count} files")
        else:
            print(f"{dir_name}: Directory not found")


def check_single_sample(data_path, sample_idx=0):
    """Kiểm tra chi tiết 1 sample"""
    pairs_file = f"{data_path}/train_pairs.txt"
    dataset = VTONDataset(root=data_path, pairs_txt=pairs_file, size=512)

    sample = dataset[sample_idx]

    print(f"\nSAMPLE {sample_idx} DETAILS:")
    print("-" * 30)
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(
                f"{key}: {value.shape} | min: {value.min():.3f} | max: {value.max():.3f}"
            )
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    data_path = "converted_train_data"  # Đường dẫn dataset của bạn

    # Hiển thị 10 samples
    visualize_dataset_samples(data_path, num_samples=10)

    # Kiểm tra chi tiết sample đầu tiên
    check_single_sample(data_path, sample_idx=0)
