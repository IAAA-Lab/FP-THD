import os
import json
import pickle
import random

def split_data_and_prepare_labels(data_path, train_ratio=0.9):
    """
    Splits image-text pairs into train and validation sets without copying images.
    Generates train.ln, val.ln, and labels.pkl.
    """
    lines_folder = os.path.join(data_path, "lines")
    train_list_file = os.path.join(data_path, "train.ln")
    val_list_file = os.path.join(data_path, "val.ln")
    labels_pkl_file = os.path.join(data_path, "labels.pkl")

    # List all image-text pairs
    image_files = [f for f in os.listdir(lines_folder) if f.endswith((".jpg", ".png", ".jpeg"))]
    txt_files = {os.path.splitext(f)[0]: f for f in os.listdir(lines_folder) if f.endswith(".txt")}

    # Match images with their corresponding text files and filter out empty text files
    valid_pairs = []
    labels = {"train": {}, "val": {}}
    charset = set()

    for img in image_files:
        img_name = os.path.splitext(img)[0]  # Get the base name without extension
        txt_file = txt_files.get(img_name)

        if txt_file:
            txt_path = os.path.join(lines_folder, txt_file)
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            if text:  # Only include non-empty text files
                valid_pairs.append(img)
                charset.update(text)  # Collect unique characters

    # Shuffle and split into train and validation sets
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * train_ratio)
    train_files = valid_pairs[:split_idx]
    val_files = valid_pairs[split_idx:]

    # Save image lists
    with open(train_list_file, "w") as f:
        f.write("\n".join(train_files))

    with open(val_list_file, "w") as f:
        f.write("\n".join(val_files))

    # Save labels.pkl
    for img in train_files:
        txt_path = os.path.join(lines_folder, f"{os.path.splitext(img)[0]}.txt")
        with open(txt_path, 'r', encoding='utf-8') as f:
            labels["train"][img] = {"text": f.read().strip()}

    for img in val_files:
        txt_path = os.path.join(lines_folder, f"{os.path.splitext(img)[0]}.txt")
        with open(txt_path, 'r', encoding='utf-8') as f:
            labels["val"][img] = {"text": f.read().strip()}

    with open(labels_pkl_file, "wb") as f:
        pickle.dump({"ground_truth": labels, "charset": sorted(list(charset))}, f)

    print(f"✅ Dataset split and labels prepared in: {data_path}")
    print(f"📄 Train images: {len(train_files)}, Val images: {len(val_files)}")
    print(f"🔤 Unique characters: {len(charset)}")

# Run the function
data_path = "./MDM"
split_data_and_prepare_labels(data_path)
