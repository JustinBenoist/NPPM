import math
import argparse
from typing import List
import pickle
import matplotlib
matplotlib.rcParams['text.color'] = "black"
import numpy as np
import pyexr
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()  # In case something was left over
import mitsuba as mi 


from integrators.PPM import PPMIntegrator
from integrators.NPPM import NPPMIntegrator
from integrators.Model import DPMGaussianAnisoOptGrid, DPMGaussianIsoOptGrid, PhotonEncoder, DPMGaussianAnisoOptGridNoEncoder


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
    parser.add_argument('--ref', default=None, type=str, help='reference image if we have to compute error')
    parser.add_argument('--outfile', type=str, default="output/test.exr", help='output file')
    parser.add_argument('--error_out', type=str, default="output/error.pkl", help='error tracking file')
    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint file')
    parser.add_argument('--model', type=str, default='', help='model .pth path')
    parser.add_argument('--encoder', type=str, default=None, help='photon encoder .pth path')
    parser.add_argument('--ppi', type=int, default=100000, help='photons per iteration')
    parser.add_argument('--iter', type=int, default=100, help='number of iterations')
    parser.add_argument('--radius', type=float, default=0, help='inital radius in pixel footprint scale, or in world space if --no_ray_diff is active')
    parser.add_argument('--neighbors', type=int, default=200, help='number of max neighbors in radius search')
    parser.add_argument('--seed', type=int, default=0, help='seed to use')
    parser.add_argument('--update_proportion', type=float, default=0.1, help='proportion of gather points used for updating grid')
    parser.add_argument('--stop_grid', type=int, default=50, help='limit iteration count to stop updating the grid')
    parser.add_argument('--res_grid', type=int, default=256, help='resolution of the DCV grid')
    parser.add_argument('--time_limit', type=float, default=None, help='time budget, defaults to unlimited')
    parser.add_argument('--build_guiding', type=int, default=None, help='nb of iterations for build the guding histogram, defaults to no guiding')
    
    parser.add_argument('--dcv_size', type=int, default=32, help='Deep Context Vector size (power of 2)')
    parser.add_argument('--large', action="store_true", default=False, help="indicates if large model is used")
    parser.add_argument('--classic', action="store_true", default=False, help="indicates if original DPM model is used")
    parser.add_argument('--start', type=int, default=0, help='starting iteration, deprecated')
    parser.add_argument('--ppm', action="store_true", default=False, help="indicates if non stochastic PPM is used (fixes the gather points)")
    parser.add_argument('--isotropic', action="store_true", default=False, help="indicates if isotropic gaussian is used")
    parser.add_argument('--all_caustics', action="store_true", default=False, help="indicates if we use indirect caustic illumination photons")
    parser.add_argument('--no_ratio_matches', action="store_true", default=False, help="indicates if we use ratio of matches as network input")
    parser.add_argument('--knn_mode', action="store_true", default=False, help="indicates if we use KNN search instead of radius search to gather photons")
    parser.add_argument('--no_ray_diff', action="store_true", default=False, 
                        help="indicates if we use ray differentials to initialize the radii. Pixel footprint is controllable using --radius")
    parser.add_argument('--cppm', action="store_true", default=False, help="indicates if we use CPPM")
    parser.add_argument('--beta', type=float, default=1.2, help='parameter for CPPM radius redution, only for NPPM^beta')
    parser.add_argument('--k', type=float, default=0.8, help='parameter for CPPM radius redution, only for NPPM^beta')
    parser.add_argument('--min_photon', type=int, default=10, help='parameter for CPPM radius redution, only for NPPM^beta')
    parser.add_argument('--cut_radius', type=float, default=1.0, help='Ray differential scale under which we use box kernel')
    opt = parser.parse_args()
    print(opt)
    
    PHOTONS_PER_ITER = opt.ppi
    N_ITER = opt.iter
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Model running on {device}")
    MAX_PHOTONS = opt.neighbors
    OUT = opt.outfile
    NEIGHBORS = opt.neighbors
    PHOTONS_ITER = opt.ppi
    RADIUS = opt.radius
    ALPHA = 2 / 3
    USE_MASK = True
    SIZE = "large" if opt.large else "small"
    GAUSSIAN = not opt.classic
    APAM = True
    STOCHASTIC = not (opt.ppm)
    N_SPLIT = 1 if opt.model == "" else math.ceil(opt.update_proportion * 10)

    USE_RAY_DIFF = not opt.no_ray_diff
    
    # Initialize scene
    scene: mi.Scene = mi.load_file(opt.scene)

    if opt.ref is not None:
        gt = pyexr.read(opt.ref)
        gt = torch.nan_to_num(torch.from_numpy(gt).reshape(1, gt.shape[0] * gt.shape[1], 3).to(device), 0.0)
    else:
        gt = None
    params = mi.traverse(scene)
    camera: mi.ProjectiveCamera = scene.sensors()[0]
    resolution = camera.film().size()

    radius = RADIUS
    if opt.checkpoint != "":
        radius = opt.radius * torch.sqrt(torch.prod(torch.tensor([(k + ALPHA) / k for k in range(1, opt.start + 1)], device=device)) / (opt.start + 1))
        print(f"STARTING RADIUS = {radius}")
        
    encoder = PhotonEncoder(NEIGHBORS, opt.dcv_size, 2, use_mask=USE_MASK, n_hidden_pred=3)
    if opt.encoder is not None:
        encoder.load_state_dict(torch.load(opt.encoder, map_location=device))
    encoder = encoder.to(device)

    
    if opt.model == "":
        ppm = PPMIntegrator(PHOTONS_ITER, N_ITER, device, radius, start_iter=0, n_split=N_SPLIT, apam=APAM, stochastic=STOCHASTIC, seed=opt.seed,
                            pure_caustic=(not opt.all_caustics), knn_mode=opt.knn_mode, use_ray_diff=(not opt.no_ray_diff), is_cppm=opt.cppm, time_limit=opt.time_limit,
                            build_guiding=opt.build_guiding)
        checkpoint_sppm = torch.from_numpy(pyexr.read(opt.checkpoint)).to(device=device).reshape(resolution.y * resolution.x, 3) if opt.checkpoint != "" else None
        ppm_test, error_ppm = ppm.run(scene, OUT, gt=gt, checkpoint=checkpoint_sppm)
        if gt is not None:
            save_error(opt.checkpoint, error_ppm, opt.error_out)
        ppm_estimate_test2 = ppm_test.reshape(resolution.y, resolution.x, 3)
        pyexr.write(OUT, ppm_estimate_test2.cpu().detach().numpy().astype(np.float32))
    else:
        size = "large" if opt.large else "small"
        if opt.isotropic:
            network = DPMGaussianIsoOptGrid(NEIGHBORS, opt.dcv_size, size=SIZE, ratio_matches=(not opt.no_ratio_matches), 
                                            stop=opt.stop_grid).to(device)
        else:
            if opt.encoder is not None:
                network = DPMGaussianAnisoOptGrid(encoder, NEIGHBORS, opt.dcv_size, size=SIZE, ratio_matches=(not opt.no_ratio_matches), 
                                                stop=opt.stop_grid).to(device)
            else:
                network = DPMGaussianAnisoOptGridNoEncoder(NEIGHBORS, opt.dcv_size, size=SIZE, ratio_matches=(not opt.no_ratio_matches), 
                                                stop=opt.stop_grid).to(device)
        GAUSSIAN = True
        network.load_state_dict(torch.load(opt.model, map_location=device))
        network = network.to(device)
        
        dppm = NPPMIntegrator(network, PHOTONS_ITER, N_ITER, device, radius, NEIGHBORS, start_iter=opt.start, n_split=N_SPLIT, apam=APAM, stochastic=STOCHASTIC, 
                              seed=opt.seed, pure_caustic=(not opt.all_caustics), knn_mode=opt.knn_mode, ratio_matches=(not opt.no_ratio_matches), gaussian=GAUSSIAN, 
                              use_ray_diff=(not opt.no_ray_diff), build_guiding=opt.build_guiding, use_cppm_scheme=opt.cppm, beta=opt.beta, k=opt.k, min_photon=opt.min_photon, 
                              cut_radius=opt.cut_radius, update_proportion=opt.update_proportion, res_grid=opt.res_grid, time_limit=opt.time_limit)
        checkpoint_sdppm = torch.from_numpy(pyexr.read(opt.checkpoint)).to(device=device).reshape(resolution.y * resolution.x, 3) if opt.checkpoint != "" else None
        dppm_test, error_dppm = dppm.run(scene, OUT, gt=gt, checkpoint=checkpoint_sdppm)
        if gt is not None:
            save_error(opt.checkpoint, error_dppm, opt.error_out)
        dppm_estimate_test = dppm_test.reshape(resolution.y, resolution.x, 3)
        pyexr.write(OUT, dppm_estimate_test.cpu().detach().numpy().astype(np.float32))