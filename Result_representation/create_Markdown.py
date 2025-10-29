import os, re
import glob
import xml.etree.ElementTree as ET

NAMESPACE = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'}

def get_x_from_points(points_str):
    points = points_str.split()
    xs = [int(pt.split(',')[0]) for pt in points]
    return sum(xs) / len(xs)

def get_min_max_x(points_str):
    points = points_str.split()
    xs = [int(pt.split(',')[0]) for pt in points]
    return min(xs), max(xs)

def extract_text_lines(region):
    lines = []
    for textline in region.findall('pc:TextLine', NAMESPACE):
        unicode_elem = textline.find('.//pc:Unicode', NAMESPACE)
        if unicode_elem is not None and unicode_elem.text:
            clean = unicode_elem.text.strip()
            if clean:
                lines.append(clean)
    return lines

def count_textlines(region):
    return len(region.findall('pc:TextLine', NAMESPACE))

def find_best_divider(x_avgs):
    if not x_avgs:
        return 1000  # fallback
    x_avgs = sorted(x_avgs)
    max_gap = 0
    divider = x_avgs[0]
    for i in range(len(x_avgs) - 1):
        gap = x_avgs[i+1] - x_avgs[i]
        if gap > max_gap:
            max_gap = gap
            divider = (x_avgs[i] + x_avgs[i+1]) / 2
    return divider

def get_baseline_center_x(textline):
    baseline = textline.find('pc:Baseline', NAMESPACE)
    if baseline is not None and 'points' in baseline.attrib:
        return get_x_from_points(baseline.attrib['points'])
    # fallback: use coords center if baseline missing
    coords = textline.find('pc:Coords', NAMESPACE)
    if coords is not None:
        return get_x_from_points(coords.attrib['points'])
    return None

def parse_page(xml_file, min_textlines_required=4):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    page = root.find('pc:Page', NAMESPACE)

    try:
        page_width = int(page.attrib['imageWidth'])
    except (KeyError, ValueError):
        raise ValueError(f"Missing or invalid imageWidth in file: {xml_file}")

    regions = page.findall('pc:TextRegion', NAMESPACE)

    # Collect average X for valid regions (with enough lines)
    x_avgs = []
    for region in regions:
        if count_textlines(region) >= min_textlines_required:
            coords = region.find('pc:Coords', NAMESPACE)
            if coords is not None:
                x_avg = get_x_from_points(coords.attrib['points'])
                x_avgs.append(x_avg)

    divider = find_best_divider(x_avgs)

    left_column = []
    right_column = []

    # Now assign each TextLine individually based on baseline center X
    for region in regions:
        for textline in region.findall('pc:TextLine', NAMESPACE):
            baseline_x = get_baseline_center_x(textline)
            if baseline_x is None:
                continue
            unicode_elem = textline.find('.//pc:Unicode', NAMESPACE)
            if unicode_elem is None or not unicode_elem.text or len(unicode_elem.text.strip()) == 0:
                continue
            text = unicode_elem.text.strip()

            if baseline_x < divider:
                left_column.append(text)
            else:
                right_column.append(text)

    return left_column, right_column

def extract_number(filename):
    match = re.search(r'(\d+)', os.path.basename(filename))
    return int(match.group(1)) if match else 0
    
    
def generate_md_for_pages(xml_files):
    md = []
    xml_files = sorted(xml_files, key=extract_number)
    for i, xml_file in enumerate((xml_files), 1):
        left_col, right_col = parse_page(xml_file)
        
        leaf_num = (i + 1) // 2  # integer division to get the leaf number
        
        if i % 2 == 1:
            page_label = f"Folio {leaf_num}, recto"
        else:
            page_label = f"Folio {leaf_num}, verso"
        
        md.append(f"# {page_label}\n")
        
        md.append("## Anotacion manuscrita\n")
        md.append("## Cabecera\n")

        md.append("## Columna Izquierda\n")
        if left_col:
            for idx, line in enumerate(left_col, 1):
                md.append(f"{line}<br>")
        else:
            md.append("- (vacío)")

        md.append("\n## Columna Derecha\n")
        if right_col:
            for idx, line in enumerate(right_col, 1):
                md.append(f"{line}<br>")
        else:
            md.append("- (vacío)")

        md.append("## Reclamo")
        md.append("## Cuaderno")
        
        # Add "Pie" only if verso page
        if i % 2 == 0:
            md.append("## Pie")
            
        md.append("\n---\n")

    return "\n".join(md)


if __name__ == "__main__":
    xml_files = glob.glob("./text_xmls/*with_text.xml")  # Adjust as needed  md/*.xml
    md_text = generate_md_for_pages(xml_files)
    with open("Miguel_del_Molino_transcripcion.md", "w", encoding="utf-8") as f:
        f.write(md_text)

