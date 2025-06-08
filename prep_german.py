from datasets import load_dataset
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import pickle

dataset = load_dataset('fhswf/german_handwriting')['train']
output_root = os.path.expanduser("~/task3DLP/Datasets/GERMAN_lines")  # Changed to home directory

# First ensure the base directory exists
try:
    os.makedirs(output_root, exist_ok=True)
except PermissionError:
    print(f"Error: Cannot create directory {output_root}")
    print("Please choose a different output location where you have write permissions")
    exit(1)

# Prepare splits
all_data = list(dataset)
train_data, temp_data = train_test_split(all_data, test_size=0.2, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

splits = {
    "train": train_data,
    "valid": valid_data,
    "test": test_data,
}

# Collect all characters to build the charset
charset = set()
for item in all_data:
    if item["text"] is not None:
        charset.update(set(item["text"]))
charset = sorted(list(charset))

for split_name, data_split in splits.items():
    output_dir = os.path.join(output_root, split_name)
    img_dir = output_dir
    
    try:
        os.makedirs(img_dir, exist_ok=True)
    except PermissionError as e:
        print(f"Error creating directory {img_dir}: {e}")
        continue
    
    # Create ground truth dictionary
    gt = {}
    with open(os.path.join(output_dir, "lines.txt"), "w", encoding="utf-8") as f:
        for idx, item in enumerate(data_split):
            img = item["image"]
            label = item["text"] if item["text"] is not None else "[UNK]"
            img_path = f"{idx}.png"
            img.save(os.path.join(img_dir, img_path))
            f.write(f"{img_path}\t{label}\n")
            gt[img_path] = label

    # Save labels.pkl
    with open(os.path.join(output_dir, "labels.pkl"), "wb") as f:
        pickle.dump({
            "ground_truth": {split_name: gt},
            "charset": charset
        }, f)

# Create combined labels.pkl
all_gt = {}
for split_name in splits.keys():
    split_dir = os.path.join(output_root, split_name)
    try:
        with open(os.path.join(split_dir, "labels.pkl"), "rb") as f:
            data = pickle.load(f)
            all_gt[split_name] = data["ground_truth"][split_name]
    except FileNotFoundError:
        continue

with open(os.path.join(output_root, "labels.pkl"), "wb") as f:
    pickle.dump({
        "ground_truth": all_gt,
        "charset": charset
    }, f)

print(f"Dataset successfully prepared in {output_root}")