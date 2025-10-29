import os
import json
import pickle

def split_data_and_prepare_labels_from_txt(data_path, train_file="Benthman/train.txt", val_file="Benthman/val.txt", test_file="Benthman/test.txt"):
    """
    Splits image-text pairs into train, validation, and test sets based on file lists provided in text files.
    Generates train.ln, val.ln, test.ln, and labels.pkl.
    """
    lines_folder = os.path.join(data_path, "lines")
    train_list_file = os.path.join(data_path, "train.ln")
    val_list_file = os.path.join(data_path, "val.ln")
    test_list_file = os.path.join(data_path, "test.ln")
    labels_pkl_file = os.path.join(data_path, "labels.pkl")

    # Read the list of filenames for train, validation, and test from the provided text files
    with open(train_file, 'r', encoding='utf-8') as f:
        train_files = [line.strip() for line in f.readlines()]
    
    with open(val_file, 'r', encoding='utf-8') as f:
        val_files = [line.strip() for line in f.readlines()]

    with open(test_file, 'r', encoding='utf-8') as f:
        test_files = [line.strip() for line in f.readlines()]

    # Match images with their corresponding text files
    txt_files = {os.path.splitext(f)[0]: f for f in os.listdir(lines_folder) if f.endswith(".txt")}
    valid_pairs = []
    labels = {"train": {}, "val": {}, "test": {}}
    charset = set()

    # Prepare labels for each dataset
    for img_list, dataset in zip([train_files, val_files, test_files], ["train", "val", "test"]):
        for img in img_list:
            # Ensure the filename has the .png extension
            img_with_extension = img + ".png" if not img.endswith(".png") else img
            txt_file = txt_files.get(os.path.splitext(img_with_extension)[0])
            if txt_file:
                txt_path = os.path.join(lines_folder, txt_file)
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()

                if text:  # Only include non-empty text files
                    valid_pairs.append(img_with_extension)  # Save the image with .png extension
                    labels[dataset][img_with_extension] = {"text": text}
                    charset.update(text)  # Collect unique characters

    # Save image lists for train, val, and test sets
    with open(train_list_file, "w") as f:
        f.write("\n".join([img + ".png" if not img.endswith(".png") else img for img in train_files]))

    with open(val_list_file, "w") as f:
        f.write("\n".join([img + ".png" if not img.endswith(".png") else img for img in val_files]))

    with open(test_list_file, "w") as f:
        f.write("\n".join([img + ".png" if not img.endswith(".png") else img for img in test_files]))

    # Save labels.pkl
    with open(labels_pkl_file, "wb") as f:
        pickle.dump({"ground_truth": labels, "charset": sorted(list(charset))}, f)

    print(f"Dataset processed and labels prepared in: {data_path}")
    print(f"Train images: {len(train_files)}, Val images: {len(val_files)}, Test images: {len(test_files)}")
    print(f"Unique characters: {len(charset)}")

# Run the function
data_path = "./Benthman"
split_data_and_prepare_labels_from_txt(data_path)
