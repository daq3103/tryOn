#!/usr/bin/env python3
"""
Script test để kiểm tra xem code có thể chạy được không
"""

import torch
import os
import tempfile
import shutil

def test_imports():
    """Test import các modules"""
    print("Testing imports...")
    try:
        from data.datasets import VTONDataset
        from models.zero_shot_tryon import ZeroShotTryOn
        from conditioning.multi_source_attn import MultiSourceAttnProcessor
        from conditioning.fuse import GatedFusion
        from modules.encoders.promt_encoder import TextEncoder
        from modules.encoders.garment_encoder import GarmentEncoder
        from modules.encoders.person_encoder import PersonEncoder
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_creation():
    """Test tạo model"""
    print("\nTesting model creation...")
    try:
        from diffusers import UNet2DConditionModel
        from models.zero_shot_tryon import ZeroShotTryOn
        
        # Tạo UNet nhỏ để test
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
        
        model = ZeroShotTryOn(unet)
        print("✅ Model creation successful!")
        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_forward_pass():
    """Test forward pass"""
    print("\nTesting forward pass...")
    try:
        from diffusers import UNet2DConditionModel
        from models.zero_shot_tryon import ZeroShotTryOn
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Tạo UNet nhỏ
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
        
        model = ZeroShotTryOn(unet)
        model.to(device)
        
        # Test data
        batch_size = 1
        imgs = torch.randn(batch_size, 3, 512, 512).to(device)
        person_cond = torch.randn(batch_size, 3+18+20, 512, 512).to(device)
        garment_img = torch.randn(batch_size, 3, 512, 512).to(device)
        prompts = ["a person wearing a red shirt"]
        
        # Forward pass
        loss = model(imgs, person_cond, garment_img, prompts)
        print(f"✅ Forward pass successful! Loss: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"❌ Forward pass error: {e}")
        return False

def test_dataset():
    """Test dataset creation"""
    print("\nTesting dataset...")
    try:
        # Tạo temporary data
        with tempfile.TemporaryDirectory() as temp_dir:
            # Tạo thư mục cần thiết
            os.makedirs(os.path.join(temp_dir, "person"), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, "garment"), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, "target"), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, "pose"), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, "parsing"), exist_ok=True)
            
            # Tạo file pairs
            pairs_file = os.path.join(temp_dir, "train_pairs.txt")
            with open(pairs_file, 'w') as f:
                f.write("person1.jpg\tgarment1.jpg\t\"a person wearing a red shirt\"\n")
            
            # Tạo ảnh test (1x1 pixel)
            from PIL import Image
            test_img = Image.new('RGB', (1, 1), color='red')
            test_img.save(os.path.join(temp_dir, "person", "person1.jpg"))
            test_img.save(os.path.join(temp_dir, "garment", "garment1.jpg"))
            test_img.save(os.path.join(temp_dir, "target", "person1_garment1.jpg"))
            
            # Test dataset
            from data.datasets import VTONDataset
            dataset = VTONDataset(root=temp_dir, pairs_txt="train_pairs.txt", size=64)
            sample = dataset[0]
            
            print(f"✅ Dataset test successful! Sample keys: {list(sample.keys())}")
            return True
    except Exception as e:
        print(f"❌ Dataset test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running tests for Zero-Shot VTON...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_creation,
        test_forward_pass,
        test_dataset,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Code should be ready for training.")
    else:
        print("⚠️  Some tests failed. Please fix the issues before training.")
    
    return passed == total

if __name__ == "__main__":
    main()
