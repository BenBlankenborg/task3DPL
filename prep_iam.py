import os
import shutil
import pickle
from sklearn.model_selection import train_test_split
from collections import defaultdict
from pathlib import Path

# ==== PATH CONFIG ====
img_dir = "/home3/s4895606/task3DLP/Datasets/IAM-data/IAM-data/img"
gt_file = "/home3/s4895606/task3DLP/Datasets/IAM-data/IAM-data/iam_lines_gt.txt"
output_root = "formatted/IAM_lines"

# ==== LOAD GROUND TRUTH ====
with open(gt_file, 'r') as f:
    lines = f.read().strip().split('\n\n')

samples = []
for block in lines:
    parts = block.strip().split('\n')
    if len(parts) != 2:
        continue
    filename, text = parts
    samples.append((filename.strip(), text.strip()))

# ==== SPLIT DATA ====
train_data, temp_data = train_test_split(samples, test_size=0.2, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

splits = {
    "train": train_data,
    "valid": valid_data,
    "test": test_data
}

# ==== FORMAT AND SAVE ====
gt = defaultdict(dict)
charset = set()

for split_name, split_samples in splits.items():
    split_folder = os.path.join(output_root, split_name)
    os.makedirs(split_folder, exist_ok=True)

    for idx, (orig_filename, text) in enumerate(split_samples):
        src_img_path = os.path.join(img_dir, orig_filename)
        if not os.path.exists(src_img_path):
            print(f"Warning: Missing image {orig_filename}, skipping.")
            continue

        new_img_name = f"{split_name}_{idx}.png"
        dst_img_path = os.path.join(split_folder, new_img_name)

        shutil.copy2(src_img_path, dst_img_path)

        gt[split_name][new_img_name] = {"text": text}
        charset.update(text)

# ==== SAVE LABEL FILE ====
os.makedirs(output_root, exist_ok=True)
with open(os.path.join(output_root, "labels.pkl"), "wb") as f:
    pickle.dump({
        "ground_truth": gt,
        "charset": sorted(list(charset))
    }, f)

print("✅ Formatting complete. Output saved to:", output_root)
