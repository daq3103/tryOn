import os
import shutil
from glob import glob

def convert_test_data_to_train_format(
    test_dir, output_dir, max_samples=None, add_prompt="a person trying on clothes"
):
    
    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)
    for folder in ["person", "garment", "target", "pose", "parsing"]:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    # --- Nguồn ---
    image_dir   = os.path.join(test_dir, "image")         
    cloth_dir   = os.path.join(test_dir, "cloth")         
    pose_dir    = os.path.join(test_dir, "openpose_json")
    parsing_dir = os.path.join(test_dir, "image-parse-v3")
    
    if not os.path.exists(image_dir):
        print(f"❌ {image_dir} không tồn tại")
        return
    if not os.path.exists(cloth_dir):
        print(f"⚠️  {cloth_dir} không tồn tại (sẽ bỏ qua garment)")

    # Helper: liệt kê ảnh nhiều đuôi
    def list_imgs(folder):
        exts = ["*.jpg", "*.jpeg", "*.png"]
        files = []
        for e in exts:
            files.extend(glob(os.path.join(folder, e)))
        return sorted(files)

    # Danh sách file
    person_paths = list_imgs(image_dir)
    cloth_paths  = list_imgs(cloth_dir) if os.path.exists(cloth_dir) else []

    person_files = [os.path.basename(p) for p in person_paths]
    cloth_files  = [os.path.basename(p) for p in cloth_paths]

    if max_samples is not None:
        person_files = person_files[:max_samples]

    print(f"Found {len(person_files)} person images, {len(cloth_files)} cloth images")

    pairs = []
    copied = 0

    for person_file in person_files:
        person_id, person_ext = os.path.splitext(person_file)

        # --- copy person (-> person/) ---
        src_person = os.path.join(image_dir, person_file)
        dst_person = os.path.join(output_dir, "person", person_file)
        shutil.copy2(src_person, dst_person)

        # --- chọn cloth ---
        if cloth_files:
            cloth_file = person_file if person_file in cloth_files else cloth_files[copied % len(cloth_files)]
            src_cloth = os.path.join(cloth_dir, cloth_file)
            dst_cloth = os.path.join(output_dir, "garment", cloth_file)
            shutil.copy2(src_cloth, dst_cloth)
        else:
            cloth_file = person_file  # placeholder nếu thiếu cloth dir
            dst_cloth = None

        # --- target = person gốc (-> target/) ---
        # Nếu muốn giữ cùng đuôi theo person:
        cloth_id, _ = os.path.splitext(cloth_file)
        target_name = f"{person_id}_{cloth_id}{person_ext}"
        dst_target = os.path.join(output_dir, "target", target_name)
        shutil.copy2(src_person, dst_target)

        # --- pose (nếu có) ---
        pose_json_src = os.path.join(pose_dir, f"{person_id}_keypoints.json")
        if os.path.exists(pose_json_src):
            dst_pose = os.path.join(output_dir, "pose", f"{person_id}.json")
            shutil.copy2(pose_json_src, dst_pose)

        # --- parsing (nếu có) ---
        parsing_png_src = os.path.join(parsing_dir, f"{person_id}.png")
        if os.path.exists(parsing_png_src):
            dst_parsing = os.path.join(output_dir, "parsing", f"{person_id}.png")
            shutil.copy2(parsing_png_src, dst_parsing)

        # --- ghi cặp (person_file, cloth_file, prompt) ---
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
    test_dir = "/content/tryOn/viton/train"
    output_dir = "/content/tryOn/viton/converted_train_data"
    convert_test_data_to_train_format(
        test_dir=test_dir,
        output_dir=output_dir,
        max_samples=None,
        add_prompt="a person trying on clothes",
    )
