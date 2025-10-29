import os
import cv2
import xml.etree.ElementTree as ET
import argparse
import numpy as np

def crop_text_lines(xml_file_path, input_image_folder, output_folder):
    """
    Crops text lines from an image based on the coordinates in an XML file and saves
    the recognized text for each line with the line's ID as the filename.

    Args:
        xml_file_path (str): Path to the XML file containing line coordinates and text.
        input_image_folder (str): Path to the folder containing the input image.
        output_folder (str): Path to the folder where cropped line images and text
            files will be saved.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Handle namespace
    namespace = ""
    if root.tag.startswith('{'):
        uri, tag = root.tag[1:].split('}', 1)
        namespace = {'ns': uri}
        print(f"Detected namespace: {uri}")
    else:
        namespace = None

    # Helper function to find elements with namespace
    def find_element(element, path, namespace=None):
        if namespace:
            return element.find(path, namespace)
        else:
            return element.find(path)

    # Helper function to findall elements with namespace
    def findall_elements(element, path, namespace=None):
        if namespace:
            return element.findall(path, namespace)
        else:
            return element.findall(path)

    # Get the image file name from the XML
    page_element = find_element(root, ".//ns:Page", namespace)
    if page_element is not None:
        image_filename = page_element.get("imageFilename")
    else:
        print("Error: <Page> element not found in XML.")
        return

    if image_filename is None:
        print("Error: 'imageFilename' attribute not found in <Page> element.")
        return

    image_path = os.path.join(input_image_folder, image_filename)

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    # Get the base name of the image (without extension)
    image_base_name = os.path.splitext(image_filename)[0]

    # Iterate through TextLine elements
    for region in findall_elements(root, ".//ns:TextRegion", namespace):
        for line in findall_elements(region, ".//ns:TextLine", namespace):
            line_id = line.get("id")  # Get the line ID
            if not line_id:
                print("Warning: TextLine has no 'id' attribute. Skipping.")
                continue

            # Get coordinates of the line
            coords_element = find_element(line, "ns:Coords", namespace)
            if coords_element is not None:
                coords = coords_element.get("points").split()
                points = [tuple(map(int, coord.split(","))) for coord in coords]
            else:
                print(f"Warning: <Coords> element not found in TextLine {line_id}. Skipping.")
                continue

            # Extract text content
            text_equiv_element = find_element(line, "ns:TextEquiv/ns:Unicode", namespace)
            if text_equiv_element is not None:
                text_content = text_equiv_element.text
            else:
                text_content = None  # Explicitly set to None if no text content is found

            # Only proceed with cropping if there is text content
            if text_content and text_content.strip():  # Only crop if there is actual text (not empty or None)
                # Get bounding rectangle of the line
                x, y, w, h = cv2.boundingRect(np.array(points))

                # Crop the line
                cropped_line = image[y:y+h, x:x+w]

                # Construct the new filename with the image base name and line ID
                output_filename = f"{image_base_name}_page_{line_id}.jpeg"
                image_output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(image_output_path, cropped_line)

                # Save the text content to a file with the same name as the image
                text_output_path = os.path.join(output_folder, f"{image_base_name}_page_{line_id}.txt")
                with open(text_output_path, "w") as text_file:
                    text_file.write(text_content)

                print(f"Cropped and saved line {line_id} and text to {output_folder}")
            else:
                print(f"Skipped cropping for line {line_id} as no text content was found.")

def main():
    parser = argparse.ArgumentParser(description="Crop text lines from XML file and save text.")
    parser.add_argument("--xml", required=True, help="Path to the XML file.")
    parser.add_argument("--images", required=True, help="Path to the folder containing the images.")
    parser.add_argument("--output", required=True, help="Path to the output folder for cropped lines and text.")
    args = parser.parse_args()

    crop_text_lines(args.xml, args.images, args.output)

if __name__ == "__main__":
    main()

