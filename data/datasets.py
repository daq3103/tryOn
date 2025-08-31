from torch.utils.data import Dataset
import torch, cv2, json, os
import numpy as np

class VTONDataset(Dataset):
    """
    Trả về:
      person_rgb: [3,H,W]   (có thể mask vùng áo gốc)
      person_cond: [C,H,W]  (pose + parsing onehot)
      garment_img: [3,H,W]
      target_img:  [3,H,W]  (GT người mặc garment)
      prompt: str
    """
    def __init__(self, root, pairs_txt, size=512):
        self.root = root
        self.records = open(pairs_txt).read().strip().splitlines()
        self.size = size

    def __len__(self): return len(self.records)

    def _im(self, path):
        im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (self.size, self.size))
        return torch.from_numpy(im).permute(2,0,1).float()/255.

    def _load_pose(self, person_id):
        """Load pose keypoints từ file JSON hoặc numpy"""
        pose_path = os.path.join(self.root, "pose", f"{person_id[:-4]}.json")
        if os.path.exists(pose_path):
            with open(pose_path, 'r') as f:
                pose_data = json.load(f)
            # Convert pose data to heatmap (18 keypoints)
            pose_map = np.zeros((18, self.size, self.size))
            # TODO: Implement pose to heatmap conversion
            return torch.from_numpy(pose_map).float()
        else:
            # Fallback: random pose map
            return torch.randn(18, self.size, self.size)

    def _load_parsing(self, person_id):
        """Load human parsing segmentation"""
        parsing_path = os.path.join(self.root, "parsing", f"{person_id[:-4]}.png")
        if os.path.exists(parsing_path):
            parsing = cv2.imread(parsing_path, cv2.IMREAD_GRAYSCALE)
            parsing = cv2.resize(parsing, (self.size, self.size))
            # Convert to one-hot encoding (20 classes)
            parsing_onehot = np.zeros((20, self.size, self.size))
            for i in range(20):
                parsing_onehot[i] = (parsing == i).astype(np.float32)
            return torch.from_numpy(parsing_onehot).float()
        else:
            # Fallback: random parsing
            return torch.randn(20, self.size, self.size)

    def __getitem__(self, idx):
        pid, gid, prompt = self.records[idx].split('\t')
        person_rgb = self._im(os.path.join(self.root, "person", pid))
        garment_img = self._im(os.path.join(self.root, "garment", gid))
        
        # Load pose and parsing
        pose_map = self._load_pose(pid)
        parsing_map = self._load_parsing(pid)
        
        # Concatenate person condition: RGB + pose + parsing
        person_cond = torch.cat([person_rgb, pose_map, parsing_map], dim=0)
        
        target_img = self._im(os.path.join(self.root, "target", f"{pid[:-4]}_{gid}"))
        return {"person_rgb":person_rgb, "person_cond":person_cond,
                "garment_img":garment_img, "target_img":target_img,
                "prompt":prompt}
