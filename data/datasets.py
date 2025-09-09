from torch.utils.data import Dataset
import torch, cv2, json, os
import numpy as np
from PIL import Image


class VTONDataset(Dataset):

    def __init__(
        self,
        root,
        pairs_txt,
        size=512,
        num_parsing_classes=20,
        conf_thr=0.05,
        sigma=None,
        include_rgb_in_cond=False,
    ):
        self.root = root
        with open(pairs_txt, "r", encoding="utf-8") as f:
            self.records = [line.strip() for line in f if line.strip()]
        self.size = size
        self.num_parsing_classes = num_parsing_classes
        self.conf_thr = conf_thr
        self.sigma = sigma
        self.include_rgb_in_cond = (
            include_rgb_in_cond  # True -> [3+18+20], False -> [18+20]
        )

    def __len__(self):
        return len(self.records)

    def _im(self, path): 
        im = cv2.imread(path)
        if im is None:
            raise FileNotFoundError(f"Image not found: {path}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_AREA)
        ten = torch.from_numpy(im).permute(2, 0, 1).contiguous().float() / 255.0
        return ten  

    # ---------- Pose (OpenPose JSON) -> 18 heatmaps ----------
    @staticmethod
    def _gaussian_on_grid(H, W, cx, cy, sigma):
        yy, xx = np.mgrid[0:H, 0:W]
        return np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2), dtype=np.float32
        )

    def _load_pose(self, person_id):
        pose_path = os.path.join(self.root, "pose", f"{person_id[:-4]}.json") # lấy tên file json 

        with open(pose_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        people = data.get("people", [])
        if not people:
            print(f"⚠️ No people found in pose JSON: {pose_path}")
            return 
   
        flat = people[0].get("pose_keypoints_2d", [])
        # chuyển sang mảng (x,y,conf) cho 18 keypoints 
        kps = np.asarray(flat, dtype=np.float32).reshape(-1, 3)[:18] 

        img_path_guess = os.path.join(self.root, "person", person_id) 
        try:
            with Image.open(img_path_guess) as im:
                orig_w, orig_h = im.size 
        except Exception:
            # fallback: ước lượng theo max(x,y)
            max_x = max(1.0, float(np.max(kps[:, 0]))) 
            max_y = max(1.0, float(np.max(kps[:, 1])))
            orig_w, orig_h = max_x, max_y

        sx = self.size / float(orig_w) 
        sy = self.size / float(orig_h)
        sigma = (
            self.sigma if self.sigma is not None else max(2.0, self.size / 64.0 * 2.0)
        )

        pose_map = np.zeros((18, self.size, self.size), dtype=np.float32) 
        for i in range(18):
            x, y, c = kps[i]
            if c < self.conf_thr or x <= 0 or y <= 0:
                continue
            cx, cy = x * sx, y * sy  
            g = self._gaussian_on_grid(self.size, self.size, cx, cy, sigma)  
            pose_map[i] = np.maximum(pose_map[i], g) 

        return torch.from_numpy(pose_map).float() 

    # ---------- Parsing mask (PNG) -> one-hot ----------
    def _load_parsing(self, person_id):
        parsing_path = os.path.join(self.root, "parsing", f"{person_id[:-4]}.png")
        if not os.path.exists(parsing_path):
            return torch.zeros(
                self.num_parsing_classes, self.size, self.size, dtype=torch.float32
            )

        parsing = cv2.imread(parsing_path, cv2.IMREAD_GRAYSCALE)
        if parsing is None:
            return torch.zeros(
                self.num_parsing_classes, self.size, self.size, dtype=torch.float32
            )

        parsing = cv2.resize(
            parsing, (self.size, self.size), interpolation=cv2.INTER_NEAREST
        )

        K = self.num_parsing_classes
        onehot = np.zeros((K, self.size, self.size), dtype=np.float32)
        # vectorized one-hot (nhanh hơn vòng for thuần)
        for i in range(K):
            onehot[i] = parsing == i

        return torch.from_numpy(onehot).float()

    def __getitem__(self, idx):
        row = self.records[idx].split("\t")
        if len(row) < 3:
            raise ValueError(
                f"Bad line in pairs_txt (expect 3 fields): {self.records[idx]}"
            )
        pid, gid, prompt = row[0], row[1], "\t".join(row[2:]).strip().strip('"')

        person_rgb = self._im(os.path.join(self.root, "person", pid))
        # hiển thị ảnh ra để debug 
        cv2.imshow("person_rgb", person_rgb.permute(1,2,0).numpy())
        cv2.waitKey(1)

        garment_img = self._im(os.path.join(self.root, "garment", gid))
        # hiển thị ảnh ra để debug
        cv2.imshow("garment_img", garment_img.permute(1,2,0).numpy())
        cv2.waitKey(1)

        pose_map = self._load_pose(pid)
        parsing_map = self._load_parsing(pid)
        # lấy tất cả các channel của person
        person_cond = torch.cat([person_rgb, pose_map, parsing_map], dim=0)

        target_path = os.path.join(self.root, "target", f"{pid[:-4]}_{gid}")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target not found: {target_path}")
        target_img = self._im(target_path)
        # hiển thị ảnh ra để debug
        cv2.imshow("target_img", target_img.permute(1,2,0).numpy())
        while True:
            if cv2.waitKey(0) & 0xFF :
                if key == ord('p'):
                    break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                exit(0)
        return {
            "person_rgb": person_rgb,  # [3,H,W]
            "person_cond": person_cond,
            "garment_img": garment_img,  # [3,H,W]
            "target_img": target_img,  # [3,H,W]
            "prompt": prompt,
        }
