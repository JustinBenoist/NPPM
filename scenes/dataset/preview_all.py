import os
from tqdm import tqdm
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import pyexr

if __name__ == "__main__":
    current_directory = os.getcwd()
    
    xml_files = sorted([file for file in os.listdir(current_directory) if file.endswith(".xml")])
    for i in tqdm(range(len(xml_files)), desc="Computing previews"):
        file_name = xml_files[i]
        scene = mi.load_file(file_name)
        img = mi.render(scene)
        pyexr.write(os.path.join(current_directory, f"previews/preview_{i}.exr"), img.numpy().reshape((256, 256, 3)))