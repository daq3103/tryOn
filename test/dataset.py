import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import VTONDataset
from torch.utils.data import DataLoader

# Tạo dataset với relative path
dataset = VTONDataset(
    root="converted_train_data/",           # Remove ./ prefix
    pairs_txt="converted_train_data/train_pairs.txt", 
    size=256,                            
    num_parsing_classes=20,              
    conf_thr=0.05,                       
    sigma=None,                          
    include_rgb_in_cond=True             
)

print(f"Dataset size: {len(dataset)}")

# Lấy 1 sample
sample = dataset[6]
print(f"Person RGB shape: {sample['person_rgb'].shape}")       
print(f"Person cond shape: {sample['person_cond'].shape}")     
print(f"Garment shape: {sample['garment_img'].shape}")         
print(f"Target shape: {sample['target_img'].shape}")           
print(f"Prompt: {sample['prompt']}")

# Tạo DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

# Iterate qua batches
for batch in dataloader:
    print(f"Batch person_rgb: {batch['person_rgb'].shape}")     
    print(f"Batch person_cond: {batch['person_cond'].shape}")   
    print(f"Batch garment: {batch['garment_img'].shape}")       
    print(f"Batch target: {batch['target_img'].shape}")         
    print(f"Batch prompts: {batch['prompt']}")                  
    break