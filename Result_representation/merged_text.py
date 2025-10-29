import os
import re

# Input folder and output file
folder_path = './text_xmls/'
output_file = 'merged_pages_Miguel_book.txt'

# Regex to match and extract number from filenames
pattern = re.compile(r'Miguel_Del_Molino_book_page_(\d+)\.txt')

# Find all matching files and extract their numbers
files = []
for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        page_num = int(match.group(1))
        files.append((page_num, filename))

# Sort files by the number extracted from the filename
files.sort()

# Write to output file with sequential page headers
with open(output_file, 'w', encoding='utf-8') as outfile:
    for index, (original_page, filename) in enumerate(files, start=1):
        file_path = os.path.join(folder_path, filename)
        outfile.write(f"\n--------------------- page {index} --------------------\n\n")
        with open(file_path, 'r', encoding='utf-8') as infile:
            outfile.write(infile.read())
            outfile.write('\n')
