# Zero-Shot Virtual Try-On with Diffusion Models

Dự án này triển khai một hệ thống Virtual Try-On sử dụng Zero-Shot Learning với Diffusion Models.

## 🚀 Tính năng

- **Zero-Shot Learning**: Không cần fine-tune trên từng garment cụ thể
- **Multi-Source Attention**: Kết hợp text, garment, và person conditioning
- **Gated Fusion**: Tự động cân bằng trọng số giữa các nguồn thông tin
- **Post-hoc Refinement**: Tinh chỉnh kết quả cuối cùng

## 📁 Cấu trúc dự án

```
tryon/
├── conditioning/          # Attention processors và fusion
├── data/                 # Dataset và data loading
├── models/               # Model chính và losses
├── modules/              # Encoders cho text, garment, person
├── train_zero_shot.py    # Training script
├── inference.py          # Inference script
└── requirements.txt      # Dependencies
```

## 🛠️ Cài đặt

```bash
pip install -r requirements.txt
```

## 📊 Chuẩn bị dữ liệu

Cấu trúc thư mục dữ liệu:
```
data/
├── person/               # Ảnh người
├── garment/              # Ảnh áo/quần
├── target/               # Ground truth
├── pose/                 # Pose keypoints (JSON)
├── parsing/              # Human parsing (PNG)
└── train_pairs.txt       # File mapping
```

Format `train_pairs.txt`:
```
person1.jpg    garment1.jpg    "a person wearing a red shirt"
person2.jpg    garment2.jpg    "a person wearing blue jeans"
```

## 🧪 Testing

Trước khi training, hãy chạy test để đảm bảo code hoạt động:

```bash
python test_setup.py
```

## 🎯 Training

### Training nhanh (test concept):
```bash
python train_fast.py  # ~10-30 phút
```

### Training đầy đủ:
```bash
# Training với UNet nhỏ (nhanh hơn)
python train_zero_shot.py --small_unet --epochs 20 --batch_size 4

# Training với UNet full-size
python train_zero_shot.py --epochs 50 --batch_size 2

# Training với dữ liệu thực
python train_zero_shot.py --data_path /path/to/your/data --epochs 100
```

## 🔮 Inference

```bash
python inference.py
```

## 🧠 Kiến trúc

### Multi-Source Attention
- **Text Encoder**: CLIP text encoder
- **Garment Encoder**: ResNet50 + projection
- **Person Encoder**: CNN cho pose + parsing
- **Gated Fusion**: MLP để cân bằng trọng số

### Training Strategy
- Freeze UNet pretrained weights
- Chỉ train gating network và encoders
- Zero-shot spirit: không overfit vào garment cụ thể

## 📝 TODO

- [ ] Implement pose to heatmap conversion
- [ ] Add LPIPS perceptual loss
- [ ] Implement validation loop
- [ ] Add logging và visualization
- [ ] Optimize memory usage
- [ ] Add multi-GPU support

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

## 📄 License

MIT License
