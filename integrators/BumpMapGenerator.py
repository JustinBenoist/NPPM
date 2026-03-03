import argparse
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from perlin_noise import PerlinNoise # pip install perlin_noise

def generate_bumpmap(outfile: str, n: int, res: int, octave: int):
    xpix, ypix = res, res
    try:
        os.makedirs(outfile)
    except OSError:
        pass

    for i in tqdm(range(n), desc="Generating bump maps", ncols=80):
        noise = PerlinNoise(octaves=octave)
        pic = [[2 * noise([j/xpix, k/ypix]) for k in range(xpix)] for j in range(ypix)]
        pic = np.array(pic)
        im = Image.fromarray((pic * 255).astype(np.uint8))
        im.save(os.path.join(outfile, f"bumpmap_{i}.png"))
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', type=str, default="../data/bumpmaps/", help='output folder')
    parser.add_argument('--n', type=int, default=5, help='number of bump maps to generate')
    parser.add_argument('--res', type=int, default=1024, help='bump map resolution')
    parser.add_argument('--octave', type=int, default=3, help='number of octaves for Perlin noise')
    opt = parser.parse_args()
    print(opt)

    generate_bumpmap(opt.outfile, opt.n, opt.res, opt.octave)