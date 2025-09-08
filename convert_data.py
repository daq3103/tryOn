import os
import shutil
from PIL import Image
import json
from glob import glob
import random


def convert_test_data_to_train_format(
    test_dir, output_dir, max_samples=None, add_prompt="a person trying on clothes"
):
    """
    Chuyển đổi data test từ format VITON/VITON-HD sang format train
    phù hợp với VTONDataset(root/{person,garment,target,pose,parsing} + train_pairs.txt)
    """
    print(f"Converting {test_dir} -> {output_dir}")

    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)
    for folder in ["person", "garment", "target", "pose", "parsing"]:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    # Nguồn
    image_dir = os.path.join(test_dir, "image")
    cloth_dir = os.path.join(test_dir, "cloth")
    pose_dir = os.path.join(test_dir, "openpose_json")
    parsing_dir = os.path.join(test_dir, "image-parse-v3")
    if not os.path.exists(parsing_dir):
        parsing_dir = os.path.join(test_dir, "image-parse")  # fallback

    # Check tồn tại person/cloth
    if not os.path.exists(image_dir):
        print(f"❌ {image_dir} không tồn tại")
        return
    if not os.path.exists(cloth_dir):
        print(f"⚠️  {cloth_dir} không tồn tại (sẽ bỏ qua garment)")

    # Danh sách ảnh
    person_files = sorted(
        [os.path.basename(p) for p in glob(os.path.join(image_dir, "*.jpg"))]
    )
    cloth_files = (
        sorted([os.path.basename(p) for p in glob(os.path.join(cloth_dir, "*.jpg"))])
        if os.path.exists(cloth_dir)
        else []
    )

    if max_samples is not None:
        person_files = person_files[:max_samples]

    print(f"Found {len(person_files)} person images, {len(cloth_files)} cloth images")

    pairs = []
    copied = 0

    for person_file in person_files:
        person_id = person_file[:-4]  # 'xxx_yy'

        # --- copy person ---
        src_person = os.path.join(image_dir, person_file)
        dst_person = os.path.join(output_dir, "person", person_file)
        shutil.copy2(src_person, dst_person)

        # --- chọn cloth ---
        if cloth_files:
            # cùng tên nếu có, không thì random/round-robin
            cloth_file = (
                person_file
                if person_file in cloth_files
                else cloth_files[copied % len(cloth_files)]
            )
            src_cloth = os.path.join(cloth_dir, cloth_file)
            dst_cloth = os.path.join(output_dir, "garment", cloth_file)
            shutil.copy2(src_cloth, dst_cloth)
        else:
            # không có cloth dir thì bỏ qua (nhưng VTONDataset sẽ lỗi nếu thiếu)
            cloth_file = person_file
            dst_cloth = None

        # --- target = person gốc ---
        target_name = f"{person_id}_{cloth_file}"
        dst_target = os.path.join(output_dir, "target", target_name)
        shutil.copy2(src_person, dst_target)  # dùng person làm GT

        # --- pose JSON (nếu có) ---
        pose_json_src = os.path.join(pose_dir, f"{person_id}_keypoints.json")
        if os.path.exists(pose_json_src):
            dst_pose = os.path.join(output_dir, "pose", f"{person_id}.json")
            shutil.copy2(pose_json_src, dst_pose)

        # --- parsing PNG (nếu có) ---
        parsing_png_src = os.path.join(parsing_dir, f"{person_id}.png")
        if os.path.exists(parsing_png_src):
            dst_parsing = os.path.join(output_dir, "parsing", f"{person_id}.png")
            shutil.copy2(parsing_png_src, dst_parsing)

        # --- ghi pairs ---
        pairs.append(f'{person_file}\t{cloth_file}\t"{add_prompt}"')
        copied += 1
        if copied % 50 == 0:
            print(f"Processed {copied} files...")

    # --- tạo train_pairs.txt ---
    pairs_file = os.path.join(output_dir, "train_pairs.txt")
    with open(pairs_file, "w", encoding="utf-8") as f:
        f.write("\n".join(pairs))

    print(f"✅ Converted {copied} samples")
    print(f"✅ Created {pairs_file}")
    print(f"✅ Data ready at {output_dir}")


if __name__ == "__main__":
    test_dir = "datasets"  # thư mục test gốc (có image/, cloth/, openpose_json/, image-parse[-v3]/)
    output_dir = "converted_train_data"  # nơi sinh ra format train

    convert_test_data_to_train_format(
        test_dir=test_dir,
        output_dir=output_dir,
        max_samples=None,  # hoặc 100 để thử nhanh
        add_prompt="a person trying on clothes",
    )

    print("\n🎯 Train ví dụ:")
    print(
        f"python train_zero_shot.py --data_path {output_dir} --batch_size 2 --image_size 256"
    )
