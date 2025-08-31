#!/usr/bin/env python3
"""
Setup script cho Kaggle Notebook
Chạy cell này đầu tiên để setup environment
"""

import os
import sys
from pathlib import Path

def setup_kaggle_environment():
    """Setup environment cho Kaggle"""
    print("🚀 Setting up Kaggle environment...")
    
    # Tạo thư mục cho dự án
    project_dir = Path("/kaggle/working/tryon")
    project_dir.mkdir(exist_ok=True)
    
    # Tạo cấu trúc thư mục
    folders = [
        "conditioning",
        "data", 
        "models",
        "modules/encoders",
        "utils"
    ]
    
    for folder in folders:
        (project_dir / folder).mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created project structure in {project_dir}")
    return project_dir

def install_dependencies():
    """Install các dependencies cần thiết"""
    print("📦 Installing dependencies...")
    
    # Install packages
    packages = [
        "diffusers>=0.21.0",
        "transformers>=4.30.0", 
        "accelerate>=0.20.0",
        "xformers>=0.0.20",
        "opencv-python>=4.8.0",
        "Pillow>=9.5.0"
    ]
    
    for package in packages:
        os.system(f"pip install {package}")
    
    print("✅ Dependencies installed!")

def create_kaggle_notebook():
    """Tạo notebook template cho Kaggle"""
    notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Zero-Shot Virtual Try-On Training on Kaggle\\n",
    "\\n",
    "## Setup Environment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 1: Setup\\n",
    "!pip install diffusers>=0.21.0 transformers>=4.30.0 accelerate>=0.20.0 xformers>=0.0.20 opencv-python>=4.8.0 Pillow>=9.5.0\\n",
    "\\n",
    "# Clone hoặc upload code\\n",
    "!git clone https://github.com/your-repo/tryon.git /kaggle/working/tryon\\n",
    "# Hoặc upload files thủ công"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 2: Test Setup\\n",
    "import sys\\n",
    "sys.path.append('/kaggle/working/tryon')\\n",
    "\\n",
    "from test_setup import main\\n",
    "main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 3: Fast Training\\n",
    "from train_fast import train_fast\\n",
    "\\n",
    "model = train_fast()\\n",
    "print(\"✅ Training completed!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 4: Full Training (optional)\\n",
    "import subprocess\\n",
    "\\n",
    "# Training với UNet nhỏ\\n",
    "!python /kaggle/working/tryon/train_zero_shot.py --small_unet --epochs 20 --batch_size 4 --image_size 256"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
    
    with open("/kaggle/working/tryon_kaggle.ipynb", "w") as f:
        f.write(notebook_content)
    
    print("✅ Created Kaggle notebook template!")

if __name__ == "__main__":
    setup_kaggle_environment()
    install_dependencies()
    create_kaggle_notebook()
    print("\n🎉 Kaggle setup completed!")
    print("📝 Next steps:")
    print("1. Upload your code files to Kaggle")
    print("2. Run the notebook cells")
    print("3. Start training!")
