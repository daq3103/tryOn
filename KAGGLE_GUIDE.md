# 🚀 Hướng dẫn Training trên Kaggle

## 📋 **Yêu cầu Kaggle**

### **GPU Requirements:**
- **Kaggle Notebooks**: T4 GPU (16GB VRAM) - **Miễn phí**
- **Kaggle Pro**: V100 GPU (32GB VRAM) - **$9.99/tháng**
- **Kaggle Pro+**: A100 GPU (80GB VRAM) - **$49.99/tháng**

### **Time Limits:**
- **Free**: 9 giờ/ngày
- **Pro**: 20 giờ/ngày
- **Pro+**: 40 giờ/ngày

## 🎯 **Setup Kaggle Notebook**

### **Bước 1: Tạo Notebook mới**
1. Vào [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "Create" → "New Notebook"
3. Chọn "GPU" accelerator
4. Đặt tên: "Zero-Shot VTON Training"

### **Bước 2: Upload Code**
Có 2 cách:

#### **Cách 1: Upload files thủ công**
1. Click "Add data" → "Upload files"
2. Upload tất cả files từ project:
   - `train_kaggle.py`
   - `train_fast.py`
   - `test_setup.py`
   - Tất cả thư mục: `models/`, `modules/`, `conditioning/`, `data/`

#### **Cách 2: Clone từ GitHub**
```python
# Cell 1: Clone repository
!git clone https://github.com/your-username/tryon.git /kaggle/working/tryon
```

## 🔧 **Kaggle Notebook Cells**

### **Cell 1: Setup Environment**
```python
# Install dependencies
!pip install diffusers>=0.21.0 transformers>=4.30.0 accelerate>=0.20.0 xformers>=0.0.20 opencv-python>=4.8.0 Pillow>=9.5.0 tqdm matplotlib

# Check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### **Cell 2: Import và Test**
```python
import sys
sys.path.append('/kaggle/working/tryon')

# Test imports
from test_setup import main
main()
```

### **Cell 3: Fast Training (Test)**
```python
# Training nhanh để test
from train_fast import train_fast
model = train_fast()
print("✅ Fast training completed!")
```

### **Cell 4: Full Kaggle Training**
```python
# Training đầy đủ với GPU optimization
from train_kaggle import train_on_kaggle
model, losses = train_on_kaggle()
```

### **Cell 5: Save Results**
```python
# Save model và results
import shutil

# Copy best model
shutil.copy('kaggle_best_model.pth', '/kaggle/working/')

# Copy training plots
shutil.copy('final_training_progress.png', '/kaggle/working/')

print("💾 Results saved to Kaggle workspace!")
```

## ⚡ **Tối ưu cho Kaggle**

### **Memory Management:**
```python
# Thêm vào training script
import gc

# Cleanup sau mỗi epoch
gc.collect()
torch.cuda.empty_cache()
```

### **Progress Tracking:**
```python
# Sử dụng tqdm cho progress bars
from tqdm import tqdm
for batch in tqdm(dataloader, desc="Training"):
    # training code
```

### **Checkpointing:**
```python
# Save checkpoints thường xuyên
if epoch % 5 == 0:
    torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
```

## 📊 **Kaggle Training Configurations**

### **Configuration 1: Fast Test (Free GPU)**
```python
# train_fast.py settings
- UNet size: 32x32
- Batch size: 4
- Epochs: 10
- Time: ~15-30 phút
```

### **Configuration 2: Medium Training (Free GPU)**
```python
# train_kaggle.py settings
- UNet size: 64x64
- Batch size: 8
- Epochs: 30
- Time: ~2-4 giờ
```

### **Configuration 3: Full Training (Pro GPU)**
```python
# train_zero_shot.py settings
- UNet size: 128x128
- Batch size: 16
- Epochs: 50
- Time: ~6-8 giờ
```

## 🎯 **Tips cho Kaggle**

### **1. Monitor GPU Usage:**
```python
# Check GPU memory
print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

### **2. Handle Timeouts:**
```python
# Save frequently
if epoch % 2 == 0:
    torch.save(model.state_dict(), f'backup_epoch_{epoch}.pth')
```

### **3. Use Mixed Precision:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(inputs)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 📈 **Expected Results**

### **Free GPU (T4):**
- **Fast training**: ✅ Hoàn thành
- **Medium training**: ✅ Hoàn thành
- **Full training**: ⚠️ Có thể timeout

### **Pro GPU (V100):**
- **Fast training**: ✅ Rất nhanh
- **Medium training**: ✅ Nhanh
- **Full training**: ✅ Hoàn thành

### **Pro+ GPU (A100):**
- **Tất cả**: ✅ Rất nhanh và ổn định

## 🚨 **Troubleshooting**

### **Out of Memory:**
```python
# Giảm batch size
batch_size = 4  # thay vì 8

# Giảm image size
image_size = 64  # thay vì 128
```

### **Timeout:**
```python
# Giảm epochs
num_epochs = 20  # thay vì 30

# Save more frequently
save_every = 2  # thay vì 5
```

### **Import Errors:**
```python
# Check path
import sys
print(sys.path)

# Add path manually
sys.path.append('/kaggle/working/tryon')
```

## 🎉 **Kết luận**

Kaggle là **lựa chọn tuyệt vời** để train dự án này:

✅ **Miễn phí GPU**  
✅ **Dễ setup**  
✅ **Progress tracking**  
✅ **Checkpointing**  
✅ **Community support**  

**Bắt đầu với `train_fast.py` để test, sau đó nâng cấp lên `train_kaggle.py`!**
