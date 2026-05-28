import os
import re

# Get the current directory
current_directory = os.getcwd()

# Find all XML files matching "scene_{N}.xml"
xml_files = [file for file in os.listdir(current_directory) if re.match(r"scene_\d+\.xml", file)]

# Sort files based on numeric value of N
xml_files.sort(key=lambda x: int(re.search(r"scene_(\d+).xml", x).group(1)))

# Rename files with sequential numbering
for index, old_name in enumerate(xml_files, start=0):
    new_name = f"scene_{index}.xml"
    old_path = os.path.join(current_directory, old_name)
    new_path = os.path.join(current_directory, new_name)
    os.rename(old_path, new_path)
    print(f"Renamed: {old_name} -> {new_name}")