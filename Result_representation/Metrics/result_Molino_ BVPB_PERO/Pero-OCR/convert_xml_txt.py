import xml.etree.ElementTree as ET
import os
from multiprocessing import Pool

def xml_to_txt(file_pair):
    xml_file, txt_file = file_pair
    
    # Register the namespace
    ET.register_namespace('', "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15")
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Define the namespace
    ns = {'ns': "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"}
    
    with open(txt_file, 'w', encoding='utf-8') as f:
        for region in root.findall('.//ns:TextRegion', ns):
            for line in region.findall('.//ns:TextLine', ns):
                unicode_elem = line.find('.//ns:Unicode', ns)
                if unicode_elem is not None and unicode_elem.text:
                    f.write(f"{unicode_elem.text}\n")
            
            f.write("\n")  # Add a blank line between regions
    
    return f"Processed {xml_file} to {txt_file}"

def process_files(directory):
    xml_files = [f for f in os.listdir(directory) if f.endswith('.xml')]
    file_pairs = [(os.path.join(directory, f), os.path.join(directory, f.replace('.xml', '.txt'))) for f in xml_files]
    
    with Pool() as pool:
        results = pool.map(xml_to_txt, file_pairs)
    
    for result in results:
        print(result)

# Usage
directory = '10_pages/'
process_files(directory)
