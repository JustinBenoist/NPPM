import os

good_pictures_idx = [1, 3, 6, 7, 15, 17, 20, 21, 28, 29, 31, 34, 38, 51, 55, 57, 58, 59, 63, 80, 81, 84, 86, 87, 88, 94, 96, 104, 112, 114, 116, 122, 125, 129, 141, 143, 144, 145, 146, 150, 157, 159, 160, 166, 170, 172, 174, 180, 184]

if __name__ == "__main__":
    print(len(good_pictures_idx))
    for filename in os.listdir('.'):
        if filename.startswith('scene_') and filename.endswith('.xml'):
            try:
                index = int(filename[6:-4])  # Extract the number from scene_{x}.xml
                if index not in good_pictures_idx:
                    os.remove(filename)
                    print(f"Deleted: {filename}")
                else:
                    print(f"Kept: {filename}")
            except ValueError:
                print(f"Skipped (invalid format): {filename}")