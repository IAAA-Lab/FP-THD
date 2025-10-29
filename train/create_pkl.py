import os
import pickle

def create_labels_pkl(directory, output_folder):
    """
    Create labels.pkl containing train, validation, test files and charset.

    Args:
        directory (str): Path to the directory containing image and text files.
        output_folder (str): Path to the folder where labels.pkl will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize variables
    data = {"train": {}, "valid": {}, "test": {}}
    charset = set()

    # Process files
    for file_name in sorted(os.listdir(directory)):
        if file_name.endswith(".txt"):
            base_name = file_name[:-4]  # Remove ".txt" extension
            image_file = f"{base_name}.jpeg"
            text_path = os.path.join(directory, file_name)
            image_path = os.path.join(directory, image_file)

            # Check if the corresponding image file exists
            if not os.path.exists(image_path):
                print(f"Warning: Image file {image_file} not found for text file {file_name}")
                continue

            # Determine dataset type (train, valid, test)
            if base_name.startswith("train_"):
                dataset = "train"
            elif base_name.startswith("valid_"):
                dataset = "valid"
            elif base_name.startswith("test_"):
                dataset = "test"
            else:
                print(f"Skipping unrecognized file {file_name}")
                continue

            # Read text and update charset
            try:
                with open(text_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    charset.update(text)  # Add characters to charset

                # Add to the dataset
                data[dataset][image_path] = text

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Save ground truth and charset to a pickle file
    with open(os.path.join(output_folder, "labels.pkl"), "wb") as f:
        pickle.dump({
            "train": data["train"],
            "valid": data["valid"],
            "test": data["test"],
            "charset": sorted(list(charset)),  # Sorted charset
        }, f)

    print(f"labels.pkl successfully created in {output_folder}")

# Example usage
directory = "./data/read2016/lines"  # Path to your dataset directory
output_folder = "./data/read2016"  # Path to save the labels.pkl file
create_labels_pkl(directory, output_folder)
