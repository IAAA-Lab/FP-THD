import os
import glob
import torch
import json
import re
import configparser
import cv2
import numpy as np
from collections import OrderedDict
from PIL import Image
import torchvision.transforms as transforms
import logging

from Layout_analysis.core.layout import PageLayout
from Layout_analysis.document_ocr.page_parser import PageParser

# Utils and model imports (adjust as needed)
from OCR.utils import utils
from OCR.model import HTR_VT

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- OCR Processor ---
class OCRProcessor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.args.seed)
        self.out_dir = os.path.join(self.args.out_dir, self.args.exp_name)
        os.makedirs(self.out_dir, exist_ok=True)
        self.logger = utils.get_logger(self.out_dir)
        self.logger.info(json.dumps(vars(self.args), indent=4))
        self.model = self._load_model()
        self.converter = self._init_converter()

    def _load_model(self):
        model = HTR_VT.create_model(nb_cls=self.args.nb_cls, img_size=self.args.img_size[::-1])
        pth_path = os.path.join(self.args.save_dir, 'best_CER.pth')
        self.logger.info(f'Loading HWR checkpoint from {pth_path}')
        if not os.path.exists(pth_path):
            raise FileNotFoundError(f"Checkpoint file not found at {pth_path}")
        ckpt = torch.load(pth_path, map_location=self.device)
        model_dict = OrderedDict()
        pattern = re.compile('module.')
        for k, v in ckpt['state_dict_ema'].items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
            else:
                model_dict[k] = v
        model.load_state_dict(model_dict, strict=True)
        return model.to(self.device)

    def _init_converter(self):
        # Use your character list here
        character_list = [' ', '!', '&', "'", '(', ')', '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '£', '§', '«', '°', '»', 'Æ', 'Ç', 'à', 'á', 'â', 'ã', 'æ', 'ç', 'è', 'é', 'ë', 'í', 'î', 'ï', 'ñ', 'ò', 'ó', 'ô', 'õ', 'ù', 'ú', 'û', 'ü', 'ā', 'Č', 'č', 'đ', 'ē', 'ę', 'ĩ', 'ň', 'œ', 'š', 'ũ', 'ž', 'ſ', '̃', 'ẽ', '—', '†', '€', '☞']
        return utils.CTCLabelConverter(character_list)

    def process_image(self, image_path):
        self.model.eval()
        transform = transforms.Compose([
            transforms.Resize(self.args.img_size[::-1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        try:
            image = Image.open(image_path).convert('L')
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                preds = self.model(image_tensor)
                _, preds_index = preds.max(2)
                preds_size = torch.IntTensor([preds.size(1)])
                preds_str = self.converter.decode(preds_index[0], preds_size)
            return preds_str[0]
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            return ""

# --- Layout Processor ---
class LayoutProcessor:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()
        self.page_parser = self._init_page_parser()

    def _load_config(self):
        config = configparser.ConfigParser()
        config.read(self.config_path)
        if 'PAGE_PARSER' in config:
            config['PAGE_PARSER']['RUN_OCR'] = 'no'
        else:
            raise KeyError("Section 'PAGE_PARSER' not found in configuration file.")
        if 'LAYOUT_PARSER_1' in config:
            config['LAYOUT_PARSER_1']['DETECT_LINES'] = 'yes'
        else:
            raise KeyError("Section 'LAYOUT_PARSER_1' not found in configuration file.")
        return config

    def _init_page_parser(self):
        return PageParser(self.config, config_path=os.path.dirname(self.config_path))

    def process_image(self, input_image_path):
        image = cv2.imread(input_image_path, 1)
        page_layout = PageLayout(id=input_image_path, page_size=(image.shape[0], image.shape[1]))
        return self.page_parser.process_page(image, page_layout)

    def save_layout_xml(self, page_layout, output_path):
        page_layout.to_pagexml(output_path)

# --- Main Pipeline ---
def smart_sorted_regions(page_layout):
    # Sort regions by the minimum x of their polygon (leftmost point)
    return sorted(
        page_layout.regions,
        key=lambda r: min(pt[0] for pt in getattr(r, 'polygon', [(0, 0)]))
    )

def smart_sorted_lines(region):
    # Sort lines by the minimum y of their polygon (topmost point)
    return sorted(
        region.lines,
        key=lambda l: min(pt[1] for pt in getattr(l, 'polygon', [(0, 0)]))
    )
class LayoutAndOCRPipeline:
    def __init__(self, config_path, ocr_args):
        self.layout_processor = LayoutProcessor(config_path)
        self.ocr_processor = OCRProcessor(ocr_args)
        self.args = ocr_args

    def process_image(self, input_image_path, cropped_lines_folder, output_xml_path):
        logger.info(f"Detecting layout for {input_image_path} ...")
        page_layout = self.layout_processor.process_image(input_image_path)
        os.makedirs(cropped_lines_folder, exist_ok=True)

        for region_idx, region in enumerate(page_layout.regions):  # NO SORTING!
            logger.info(f"Processing region {region_idx} with {len(region.lines)} lines.")
            for line_idx, line in enumerate(region.lines):  # NO SORTING!
                line_image = line.crop.astype(np.uint8)
                line_image_path = os.path.join(
                    cropped_lines_folder,
                    f"region_{region_idx}_line_{line_idx}.jpg"
                )
                cv2.imwrite(line_image_path, line_image)
                recognized_text = self.ocr_processor.process_image(line_image_path)
                logger.info(f"Recognized text for line {line_idx} in region {region_idx}: {recognized_text}")
                line.transcription = recognized_text  # Update line-by-line

        logger.info("Saving final XML...")
        self.layout_processor.save_layout_xml(page_layout, output_xml_path)
        logger.info(f"Final XML saved to {output_xml_path}.")
        return page_layout




def save_txt_from_layout(page_layout, txt_output_path):
    lines_in_order = []
    for region in page_layout.regions:  # NO SORTING!
        for line in region.lines:       # NO SORTING!
            text = (getattr(line, "transcription", "") or "").strip()
            if text:
                lines_in_order.append(text)
    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_in_order))




# --- Argument parser (adjust as needed) ---
def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Batch OCR Pipeline")
    parser.add_argument('--config-path', type=str, required=True, help='Path to pero-ocr config file')
    parser.add_argument('--image-folder', type=str, required=True, help='Folder with input images')
    parser.add_argument('--cropped-lines-folder', type=str, required=True, help='Folder for cropped lines')
    parser.add_argument('--save-dir', type=str, required=True, help='Directory with HTR model checkpoint')
    parser.add_argument('--out-dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--exp-name', type=str, default='experiment', help='Experiment name')
    parser.add_argument('--img-size', type=int, nargs=2, default=[512, 64], help='Input HTR image size (h, w)')
    parser.add_argument('--nb-cls', type=int, default=126, help='Number of classes (characters)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--image-extension', type=str, default="tif", help='Image extension (tif, jpg, png)')
    return parser.parse_args()

def main():
    args = get_args_parser()
    out_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.cropped_lines_folder, exist_ok=True)

    # Find all images in the folder
    image_paths = sorted(glob.glob(os.path.join(args.image_folder, f"*.{args.image_extension}")))

    # Initialize pipeline
    pipeline = LayoutAndOCRPipeline(config_path=args.config_path, ocr_args=args)

    for image_path in image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        cropped_lines_folder = os.path.join(args.cropped_lines_folder, base_name)
        os.makedirs(cropped_lines_folder, exist_ok=True)

        # 1. Layout XML (structure only)
        layout_xml_path = os.path.join(out_dir, f"{base_name}_layout.xml")
        page_layout = pipeline.layout_processor.process_image(image_path)
        pipeline.layout_processor.save_layout_xml(page_layout, layout_xml_path)

        # 2. OCR & ALTO/PAGE XML (with text)
        final_xml_path = os.path.join(out_dir, f"{base_name}_with_text.xml")
        page_layout_with_text =  pipeline.process_image(
    input_image_path=image_path,
    cropped_lines_folder=cropped_lines_folder,
    output_xml_path=final_xml_path
)

        # 3. TXT output (in smart reading order)
        txt_path = os.path.join(out_dir, f"{base_name}.txt")
        save_txt_from_layout(page_layout_with_text, txt_path)

        logger.info(f"Processed {image_path}: layout XML, ALTO/PAGE XML, and TXT saved.")

    logger.info(f"All images processed. Results are in {out_dir}")

if __name__ == "__main__":
    main()
