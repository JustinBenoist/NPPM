import math
from typing import Tuple, List
from tqdm import tqdm
import numpy as np
import math
import torch
import mitsuba as mi
import pyexr
import time

from integrators.Utils import radius_search_transform
from integrators.flip.flip_loss import HDRFLIPLoss
from integrators.PPM import PPMIntegrator
from integrators.Grid import Grid3D

import torch._dynamo
torch._dynamo.config.suppress_errors = True

import frnn

def box_kernel(radius):
    return (1 / (torch.pi * radius * radius)).unsqueeze(1)

class NPPMIntegrator(PPMIntegrator):
    def __init__(self, network: torch.nn.Module, photons_per_iter: int, n_iterations: int, 
                 device: str="cuda", init_radius: float=1, max_photons: int=50, 
                 start_iter: int=0, n_split: int=8, apam: bool=True, stochastic: bool=False, seed: int=0,
                 pure_caustic: bool=True, knn_mode: bool=False, use_ray_diff: bool=True, ratio_matches: bool=False, gaussian: bool = False,
                 use_cppm_scheme: bool=False, beta: float=1.5, k: float=0.6, min_photon: int=60, cut_radius: float=1.0, 
                 update_proportion: float=0.1, res_grid: int=256, time_limit: float=None) -> None:
        super().__init__(photons_per_iter, n_iterations, 
                         device, init_radius, start_iter, n_split, apam, stochastic, seed, pure_caustic, knn_mode, use_ray_diff)
        self.network = network
        self.ratio_matches = ratio_matches
        self.max_photons = max_photons
        self.gaussian = gaussian
        frnn._C.load_normalization_table(network.gaussian_integral.values.squeeze(0).squeeze(0).flatten().contiguous())
        
        self.use_cppm_scheme = use_cppm_scheme
        self.beta = beta
        self.k = k
        self.min_photon = min_photon
        self.cut_radius = cut_radius
        self.update_proportion = update_proportion
        self.res_grid = res_grid
        self.time_limit = time_limit
        
    def run(self, scene: mi.Scene, save_path: str, gt: torch.Tensor=None, checkpoint: torch.Tensor=None) -> Tuple[float | List[float]]:
        self.network.eval()
        torch.cuda.empty_cache()
        start = time.time()
        with torch.no_grad():
            # Initializing samplers
            NB_SELEC = self.n_split
            SIZE = scene.sensors()[0].film().size().x * scene.sensors()[0].film().size().y
            N = SIZE // NB_SELEC
            sampler = mi.PCG32(size=self.photons_per_iter, initstate=self.seed)
            sampler_eye = mi.PCG32(SIZE, initstate=self.seed if self.stochastic else 42)
            # Initializing search radii
            if self.use_ray_diff:
                radius = self.initialize_radius(scene, sampler_eye, scale=self.init_radius)
            else:
                radius = torch.ones(SIZE, device=self.device) * self.init_radius
            minimal_radius = self.initialize_radius(scene, sampler_eye, scale=self.cut_radius)
            error_tab = []
            n = torch.zeros(SIZE, device=self.device) + 1e-5
            alpha = 2 / 3
            cum_flux = torch.zeros((SIZE, 3), device=self.device)
            f_size = self.network.dcv_size + (1 if self.ratio_matches else 0)
            dpc = torch.zeros((N, f_size), device=self.device).detach()
            if self.network.technique == "classic":
                dpc = dpc.unsqueeze(2)
            # Eye pass
            gps = self.eye_pass(scene, sampler_eye)
            # DCV Grid
            grid = Grid3D(scene, self.device, resolution=self.res_grid, n_features=f_size)
            n_photons = 0
            sample_size = int(self.update_proportion * N)
            if self.use_cppm_scheme:
                total_count = torch.zeros(SIZE, device=self.device)
                min_photon_count = torch.zeros(SIZE, device=self.device) + self.min_photon
                k_cppm = self.k
                beta = self.beta

            nb_matches = torch.zeros((SIZE), device=self.device, dtype=torch.int)
            # Main loop
            for i in tqdm(range(self.start_iter + 1, self.start_iter + self.n_iterations + 1), desc=f"Running NPPM ({"beta" if self.use_cppm_scheme else "alpha"})"):
                if self.stochastic:
                    gps = self.eye_pass(scene, sampler_eye)
                n_photons += self.photons_per_iter
                photon_map = self.photon_pass(scene, sampler, i)
                grid_cuda = None
                for j in range(NB_SELEC):
                    idx_range = torch.arange(N * j, N * (j + 1), device=self.device)
                    subsample = torch.randperm(N, device=self.device)[:sample_size]
                    idx_range_sub = idx_range[subsample]
                    if self.knn_mode:
                        assert False, "KNN mode not implemented yet"
                    else:
                        if i < self.network.stop:
                            # Photon subset gathering for predicting the kernel
                            inputs, m, grid_cuda = radius_search_transform(photon_map, gps[idx_range_sub], radius[idx_range_sub], self.max_photons, grid_cuda)

                    if self.apam:
                        ratio = torch.tensor([(i + alpha) / (i + 1)], device=self.device)
                    else:
                        ratio = (n + alpha * m) / (n + m)

                    # Inputs for the network
                    if i < self.network.stop:
                        network_input = inputs[0].detach()[:, :, :2].permute(0, 2, 1)
                        m = m[0].detach()[...,0]
                    ratio_matches = torch.log(1 + (100 / (m + 1))).unsqueeze(1).unsqueeze(2)
                    ratio_matches = torch.clamp(ratio_matches, torch.tensor([0.0], device=self.device), torch.tensor([2.0], device=self.device))

                    ite = i
                    pred_kernel = box_kernel(radius[idx_range])
                    radius_ok = radius[idx_range] > minimal_radius[idx_range]
                    rad = radius[idx_range]
                    gather = gps[idx_range]
                    gps_sub = gps[idx_range_sub]
                    n_match = nb_matches[idx_range]
                    # Gathering all photons and evaluating the kernel in CUDA
                    if self.gaussian:
                        inference = self.network(gather, gps_sub, network_input, m, ratio_matches, grid, ite, self.device)
                        theta, scale = inference
                        points = photon_map[:, :3]
                        wo_photons = photon_map[:, 3:6]
                        normals_photons = photon_map[:, 6:9]
                        roughness_photons = photon_map[:, 9:10]
                        is_glossy_photons = photon_map[:, 10:11].to(torch.int32)
                        flux = photon_map[:, -3:]
                        query = gps[:, :3]
                        normals = gps[:, 3:6]
                        direction = gps[:, 6:9]
                        roughness = gps[:, 9:10]
                        is_glossy = gps[:, 10:11].to(torch.int32)
                        g_eta = gps[:, 11:14]
                        g_k = gps[:, 14:17]
                        albedo = gps[:, -3:].unsqueeze(2)
                        pred_flux, raw_flux, _ = frnn.frnn_grid_photon_gauss_gather(points[None], normals_photons[None], wo_photons[None], roughness_photons[None],
                                                        is_glossy_photons[None], flux[None], query[None], normals[None], direction[None],
                                                        roughness[None], is_glossy[None], g_eta[None], g_k[None], albedo[None],
                                                        scale[None], theta[None, :, None],
                                                        r=rad[None], nb_matches=n_match[None], grid=grid_cuda)

                    if not self.gaussian:
                        pred_flux /= (rad[idx_range].unsqueeze(1) * rad[idx_range].unsqueeze(1))
                    # Adding contribution
                    current_cum = cum_flux[idx_range]
                    current_cum[radius_ok] = ((current_cum[radius_ok] * (i - 1)) + (pred_flux.squeeze(0)[radius_ok] / self.photons_per_iter)) / i
                    current_cum[~radius_ok] = ((current_cum[~radius_ok] * (i - 1)) + (pred_kernel[~radius_ok] * raw_flux.squeeze(0)[~radius_ok] / self.photons_per_iter)) / i
                    cum_flux[idx_range] = current_cum

                    # Applying radius reduction schedule
                    if self.use_cppm_scheme:
                        total_count[idx_range] += n_match
                        reduce_radius = total_count[idx_range] >= min_photon_count[idx_range]
                        radius[idx_range] = torch.where(reduce_radius, radius[idx_range] * math.sqrt(k_cppm), radius[idx_range])
                        min_photon_count[idx_range] = torch.where(reduce_radius, min_photon_count[idx_range] * beta, min_photon_count[idx_range])
                        total_count[idx_range] = torch.where(reduce_radius, 0, total_count[idx_range])
                    else:
                        radius[idx_range] *= torch.sqrt(ratio)
                # Save image every 1000th iteration except if using a time budget
                if i % 1000 == 0 and self.time_limit is None:
                    pyexr.write(save_path, cum_flux.reshape(scene.sensors()[0].film().size().y, scene.sensors()[0].film().size().x, 3).cpu().detach().numpy().astype(np.float32))
                # Returns if time's out
                if self.time_limit is not None and time.time() - start > self.time_limit:
                    return cum_flux, error_tab
                if gt is not None:
                    # flip_loss = HDRFLIPLoss()
                    # estimate_reshaped = cum_flux.reshape(scene.sensors()[0].film().size().y, scene.sensors()[0].film().size().x, 3)
                    # gt_reshaped = gt.reshape(scene.sensors()[0].film().size().y, scene.sensors()[0].film().size().x, 3)
                    SMAPE = torch.mean(torch.abs(cum_flux - gt) / (cum_flux + gt + 0.0000001), dim=0).mean(dim=0).mean(dim=0).cpu().detach().numpy()
                    relMSE = torch.mean(torch.square(cum_flux - gt) / (cum_flux + gt + 0.0000001), dim=0).mean(dim=0).mean(dim=0).cpu().detach().numpy()
                    error_tab.append((time.time() - start,
                                      torch.mean(abs(cum_flux - gt), dim=0).mean(dim=0).mean(dim=0).cpu().detach().numpy(),
                                      torch.mean(torch.square(cum_flux - gt), dim=0).mean(dim=0).mean(dim=0).cpu().detach().numpy(),
                                    #   flip_loss(estimate_reshaped.unsqueeze(0).permute(0, 3, 1, 2), gt_reshaped.unsqueeze(0).permute(0, 3, 1, 2)).cpu().detach().numpy() if self.device == "cuda" else 0,
                                      SMAPE,
                                      relMSE))
        return cum_flux, error_tab
