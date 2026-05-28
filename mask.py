import argparse
from typing import List
import pickle
import matplotlib
matplotlib.rcParams['text.color'] = "black"
import numpy as np
import pyexr
import torch
import mitsuba as mi 
import drjit as dr

print(mi.variants())
mi.set_variant('cuda_ad_rgb')

def save_error(checkpoint: str, error: List[float], output: str) -> None:
    if checkpoint == "":
        with open(output, 'wb') as f:
            pickle.dump(error, f)
    else:
        with open(output, 'rb') as f:
            old_error = pickle.load(f)
        old_error += error
        with open(output, 'wb') as f:
            pickle.dump(old_error, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, help='xml scene file')
    parser.add_argument('--iter', type=int, default=100, help='number of iterations')
    parser.add_argument('--seed', type=int, default=0, help='seed to use')
    parser.add_argument('--outfile', type=str, default="mask.exr", help='output file')
    parser.add_argument('--ref', default=None, type=str, help='reference image to mask_it')
    
    opt = parser.parse_args()
    print(opt)
    
    N_ITER = opt.iter
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Model running on {device}")
    OUT = opt.outfile
    
    scene: mi.Scene = mi.load_file(opt.scene)
    
    params = mi.traverse(scene)
    camera: mi.ProjectiveCamera = scene.sensors()[0]
    resolution = camera.film().size()

    # Compute the size
    SIZE = scene.sensors()[0].film().size().x * scene.sensors()[0].film().size().y
    masks_avg = dr.zeros(mi.Float, SIZE)
    
    seed = opt.seed
    
    sampler = mi.PCG32(SIZE, initstate=seed)
    dr.make_opaque(sampler)
    
    @dr.syntax
    def render_scene(scene: mi.Scene, sampler: mi.PCG32) -> mi.Float:
        dr.make_opaque(sampler)
        
        # Render the scene
        active = mi.Bool(True)
        
        # Generate the rays
        cam_width, cam_height = 1.0, 1.0
        pixel_size_x = cam_width / resolution.y
        pixel_size_y = cam_height / resolution.x
        xx = dr.linspace(mi.Float, pixel_size_x / 2, cam_width - pixel_size_x / 2, resolution.y)
        yy = dr.linspace(mi.Float, pixel_size_y / 2, cam_height - pixel_size_y / 2, resolution.x)
        s_x = mi.PCG32(size=resolution.y, initstate=np.random.randint(12352345))
        s_y = mi.PCG32(size=resolution.x, initstate=np.random.randint(225325423))
        dr.make_opaque(s_x, s_y)
        jitter_x = (s_x.next_float32() - 0.5) * pixel_size_x
        jitter_y = (s_y.next_float32() - 0.5) * pixel_size_y
        x, y = dr.meshgrid(
            xx + jitter_x,
            yy + jitter_y
        )
        ray, _ = camera.sample_ray(0.0, sampler.next_float32(),
                                mi.Point2f(x, y),
                                mi.Point2f(0.0, 0.0),
                                active=active)
        dr.make_opaque(ray)
        
        si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)
        bsdf: mi.BSDF = si.bsdf(ray)
        # Check if gather point is on a delta BSDF surface (97 is the flag for delta BSDF) 
        active &= si.is_valid() & (((bsdf.flags() & int(mi.BSDFFlags.Delta)) != 0) | ((bsdf.flags() & int(mi.BSDFFlags.Glossy)) != 0))
        
        return dr.select(active, 1.0, 0.0)
    
    for i in range(N_ITER):
        # Render the scene
        masks = render_scene(scene, sampler)
        dr.eval(masks)
        
        # Accumulate the masks
        masks_avg += masks 
    
    masks_avg /= N_ITER
    
    # Export the mask
    img = masks_avg.numpy().reshape((resolution.x, resolution.y))
    mi.util.write_bitmap(OUT, img)
    
    if opt.ref is not None:
        gt = pyexr.read(opt.ref)
        masked_gt = gt * (1 - img.reshape((resolution.x, resolution.y, 1)))
        
        # Add _gt to the output file
        OUT = OUT.split(".")[0] + "_gt.exr"
        img = masked_gt.reshape((resolution.x, resolution.y, 3))
        mi.util.write_bitmap(OUT, img)
        print(f"Masked GT saved to {OUT}")
        
        
        