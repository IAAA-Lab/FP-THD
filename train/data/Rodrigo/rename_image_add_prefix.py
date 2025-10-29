import os
import sys

def rename_images(text_file, image_folder, prefix):
    # Open the text file and read the lines (filenames)
    with open(text_file, 'r') as f:
        image_names = f.readlines()

    # Strip any extra whitespace/newlines from filenames
    image_names = [name.strip() for name in image_names]

    # Loop over each image name
    for image_name in image_names:
        # Add the .png extension to the image name
        image_name_with_extension = f"{image_name}.png"
        
        # Construct the old image path
        old_image_path = os.path.join(image_folder, image_name_with_extension)
        
        # Check if the image exists
        if os.path.exists(old_image_path):
            # Create the new image name by adding the prefix
            new_image_name = f"{prefix}_{image_name_with_extension}"
            
            # Construct the new image path
            new_image_path = os.path.join(image_folder, new_image_name)
            
            # Rename the image
            os.rename(old_image_path, new_image_path)
            print(f"Renamed: {old_image_path} -> {new_image_path}")
        else:
            print(f"Image not found: {old_image_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python rename_images_with_prefix.py <text_file> <image_folder> <prefix>")
        sys.exit(1)
    
    text_file = sys.argv[1]
    image_folder = sys.argv[2]
    prefix = sys.argv[3]
    
    rename_images(text_file, image_folder, prefix)

