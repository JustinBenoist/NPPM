import time
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import drjit as dr
import mitsuba as mi 
import torch
import pyexr
from integrators.flip.flip_loss import HDRFLIPLoss

print(mi.variants())
mi.set_variant('cuda_ad_rgb')

from integrators.Utils import radius_search_ppm


def box_kernel(radius):
    return (1 / (torch.pi * radius * radius)).unsqueeze(1).unsqueeze(2).repeat(1, 3, 1)

@dr.syntax
def loop_eye(scene, sampler, si, ray, wi, bsdf, throughput, roughness, is_glossy, active, ctx):
    i = mi.UInt32(0)
    wi_world = mi.Vector3f(0.0)
    while active & (i < 10):
        # BSDF sampling
        bsdf_sample, weight = bsdf.sample(ctx, si, sampler.next_float32(), mi.Point2f(sampler.next_float32(), sampler.next_float32()), active)
        throughput *= weight
        wi_world = si.sh_frame.to_world(bsdf_sample.wo)
        ray = si.spawn_ray(wi_world)
        si = scene.ray_intersect(ray)
        bsdf: mi.BSDF = si.bsdf(ray)
        # We keep on bouncing as long as the BSDF is delta or glossy enough
        roughness = bsdf.eval_attribute("alpha", si).x
        roughness = dr.select(si.is_valid() & (roughness < 1000), roughness, 0.0)
        active &= si.is_valid() & (((bsdf.flags() & int(mi.BSDFFlags.Delta)) != 0) | ((((bsdf.flags() & int(mi.BSDFFlags.Glossy)) != 0) & (roughness < 0.1))))
        is_glossy = si.is_valid() & (((bsdf.flags() & int(mi.BSDFFlags.Glossy)) != 0) & (roughness > 0.1))
        wi[si.is_valid()] = si.wi
        i += 1
    return sampler, si, ray, bsdf, throughput, roughness, is_glossy, active, wi

@dr.syntax
def loop_init(scene, sampler, si, ray, bsdf, throughput, active, ctx, t):
    i = mi.UInt32(0)
    wi_world = mi.Vector3f(0.0)
    t += dr.norm(si.p - ray.o)
    while active & (i < 10):
        # BSDF sampling
        bsdf_sample, weight = bsdf.sample(ctx, si, sampler.next_float32(), mi.Point2f(sampler.next_float32(), sampler.next_float32()), active)
        throughput *= weight
        wi_world = si.sh_frame.to_world(bsdf_sample.wo)
        # Accumulating path length
        t += dr.norm(si.p - ray.o)
        ray = si.spawn_ray(wi_world)
        si = scene.ray_intersect(ray)
        bsdf: mi.BSDF = si.bsdf(ray)
        # We keep on bouncing as long as the BSDF is delta or glossy enough
        active &= si.is_valid() & (((bsdf.flags() & int(mi.BSDFFlags.Delta)) != 0) | ((bsdf.flags() & int(mi.BSDFFlags.Glossy)) != 0))
        i += 1
    return sampler, si, bsdf, throughput, active, wi_world, t

class PPMIntegrator:
    def __init__(self, photons_per_iter: int, n_iterations: int, device: str="cuda",
                 init_radius: float=0.1, start_iter: int=0, n_split: int=1, apam: bool=True, stochastic: bool=False, seed: int=0,
                 pure_caustic: bool=True, knn_mode: bool=False, use_ray_diff: bool=True, is_cppm: bool = False,
                 time_limit: float=None) -> None:
        self.photons_per_iter = photons_per_iter
        self.n_iterations = n_iterations
        self.device = device
        self.init_radius = init_radius
        self.apam = apam
        self.stochastic = stochastic
        self.start_iter = start_iter
        self.seed = seed
        self.pure_caustic = pure_caustic
        self.knn_mode = knn_mode
        self.n_split = n_split
        self.use_ray_diff = use_ray_diff
        self.is_cppm = is_cppm
        self.time_limit = time_limit
        
    def run(self, scene: mi.Scene, save_path: str, gt: torch.Tensor=None, checkpoint: torch.Tensor=None) -> Tuple[float, List[float]]:
        start = time.time()
        with dr.suspend_grad():
            with torch.no_grad():
                # Initialize samplers
                res = scene.sensors()[0].film().size().x * scene.sensors()[0].film().size().y
                sampler = mi.PCG32(size=self.photons_per_iter, initstate=self.seed)
                sampler_eye = mi.PCG32(res, initstate=self.seed if self.stochastic else 42)
                estimate = torch.zeros((res, 3), device=self.device)
                error_tab = []
                # Initialize search radii
                if self.use_ray_diff:
                    radius = self.initialize_radius(scene, sampler_eye, scale=self.init_radius)
                else:
                    radius = torch.ones(res, device=self.device) * self.init_radius
                alpha = 2 / 3
                cum_flux = torch.zeros((res, 3), device=self.device)
                if checkpoint is not None:
                    cum_flux = checkpoint
                # Eye pass
                gps = self.eye_pass(scene, sampler_eye)
                n_photons = 0
                method_name = "SPPM" if self.stochastic else "PPM"
                if self.is_cppm:
                    method_name = "CPPM"
                NB_SELEC = self.n_split
                N = res // NB_SELEC

                if self.is_cppm:
                    # WARN: This numbers are baked in the cppm CUDA code
                    SEC_U = 2
                    SEC_V = 6
                    total_counts = torch.zeros((res), device=self.device, dtype=torch.int32)
                    min_counts = torch.ones((res), device=self.device, dtype=torch.int32) * 10 
                    photon_counts = torch.zeros((res, SEC_U * SEC_V), device=self.device, dtype=torch.int32)
                else:
                    total_counts = None
                    min_counts = None
                    photon_counts = None
                # Main loop
                for i in tqdm(range(self.start_iter + 1, self.start_iter + self.n_iterations + 1), desc=f"Running {method_name} "):
                    if self.stochastic:
                        gps = self.eye_pass(scene, sampler_eye)
                    n_photons += self.photons_per_iter
                    photon_map = self.photon_pass(scene, sampler, i)
                    for j in range(NB_SELEC):
                        idx_range = torch.arange(N * j, N * (j + 1), device=self.device)
                        if self.knn_mode:
                            assert False, "KNN mode not implemented yet"
                        
                        # Photon gathering and density estimation in CUDA, CPPM radius reduction scheme is applied in the CUDA kernel
                        if self.is_cppm:
                            value, new_radii, total_counts[idx_range], min_counts[idx_range], photon_counts[idx_range], _ = radius_search_ppm(photon_map, gps[idx_range], radius[idx_range], self.is_cppm, total_counts[idx_range], min_counts[idx_range], photon_counts[idx_range])
                            radius[idx_range] = new_radii
                        else:
                            value, _, _ = radius_search_ppm(photon_map, gps[idx_range], radius[idx_range], self.is_cppm, None, None, None)
                            
                        if self.apam:
                            ratio = torch.tensor([(i + alpha) / (i + 1)], device=self.device)
                        else:
                            assert False, "PPM radii not implemented yet"

                        cum_flux[idx_range] = ((cum_flux[idx_range] * (i - 1)) + (value / self.photons_per_iter)) / i
                        estimate[idx_range] = value / self.photons_per_iter

                        if not self.is_cppm:
                            # CPPM reduces the radius by its self in the cuda kernel, not PPM
                            radius[idx_range] *= torch.sqrt(ratio)
                    if gt is not None:
                        # flip_loss = HDRFLIPLoss()
                        # estimate_reshaped = cum_flux.reshape(scene.sensors()[0].film().size().y, scene.sensors()[0].film().size().x, 3)
                        # gt_reshaped = gt.reshape(scene.sensors()[0].film().size().y, scene.sensors()[0].film().size().x, 3)
                        smape = torch.mean((abs(cum_flux - gt) / (cum_flux + gt + 1e-5)), dim=0).mean(dim=0).mean(dim=0).cpu().detach().numpy()
                        relMSE = torch.mean((torch.square(cum_flux - gt) / (cum_flux + gt + 1e-5)), dim=0).mean(dim=0).mean(dim=0).cpu().detach().numpy()
                        error_tab.append((time.time() - start,
                                    torch.mean(abs(cum_flux - gt), dim=0).mean(dim=0).mean(dim=0).cpu().detach().numpy(),
                                    torch.mean(torch.square(cum_flux - gt), dim=0).mean(dim=0).mean(dim=0).cpu().detach().numpy(),
                                    # flip_loss(estimate_reshaped.unsqueeze(0).permute(0, 3, 1, 2), gt_reshaped.unsqueeze(0).permute(0, 3, 1, 2)).cpu().detach().numpy() if self.device == "cuda" else 0,
                                    smape,
                                    relMSE))
                    # Returns if time's out
                    if self.time_limit is not None and time.time() - start > self.time_limit:
                        return cum_flux, error_tab
                    # Save image every 1000th iteration except if using a time budget
                    if i % 1000 == 0 and self.time_limit is None:
                        pyexr.write(save_path, cum_flux.reshape(scene.sensors()[0].film().size().y, scene.sensors()[0].film().size().x, 3).cpu().detach().numpy().astype(np.float32))
        return cum_flux, error_tab
    
    def photon_pass(self, scene: mi.Scene, sampler: mi.PCG32, it: int) -> torch.Tensor:
        dr.make_opaque(sampler)
        with dr.suspend_grad():
            with torch.no_grad():
                active = mi.Bool(True)
                MAX_DEPTH = 12
                # Sample rays starting from light sources
                a = sampler.next_float32()
                b = mi.Point2f(sampler.next_float32(), sampler.next_float32())
                c = mi.Point2f(sampler.next_float32(), sampler.next_float32())
                ray, weight, _ = scene.sample_emitter_ray(0.0, a, b, c,
                                                        active=active)
                
                throughput = mi.Color3f(1.0)
                photon_flux = weight.torch().to(self.device).permute(1, 0)
                photon_pos_total = []
                photon_wo_total = []
                photon_n_total = []
                photon_flux_total = []
                photon_rough_total = []
                photon_is_glossy_total = []
                masks = []
                ctx = mi.BSDFContext(mi.TransportMode.Importance)
                
                for i in range(1, MAX_DEPTH):
                    si: mi.SurfaceInteraction3f = scene.ray_intersect(ray, active=active)
                    bsdf: mi.BSDF = si.bsdf(ray)
                    not_on_delta = (bsdf.flags() & int(mi.BSDFFlags.Delta)) == 0
                    roughness = bsdf.eval_attribute("alpha", si).x
                    roughness = dr.select(si.is_valid() & (roughness < 1000), roughness, 0.0)
                    is_glossy = si.is_valid() & ((bsdf.flags() & int(mi.BSDFFlags.Glossy)) != 0) & (roughness > 0.1)
                    # Only use caustics photons (arbitrary threshold 0.1 for deciding if we bounce on glossy surface)
                    if i == 1:
                        active &= ((bsdf.flags() & int(mi.BSDFFlags.Delta)) != 0) | (((bsdf.flags() & int(mi.BSDFFlags.Glossy)) != 0))
                    if i > 1 and self.pure_caustic:
                        active &= ((bsdf.flags() & int(mi.BSDFFlags.Diffuse)) == 0) | (((bsdf.flags() & int(mi.BSDFFlags.Glossy)) == 0) & (roughness > 0.1))
                    
                    bsdf_sample, weight = bsdf.sample(ctx, si, sampler.next_float32(), 
                                                    mi.Point2f(sampler.next_float32(), sampler.next_float32()),
                                                    active=active)
                    wo_world = si.to_world(bsdf_sample.wo)
                    active &= si.is_valid()
                    
                    if i > 1:
                        photon_pos_total.append(si.p.torch().to(self.device).permute(1, 0))
                        photon_wo_total.append(si.wi.torch().to(self.device).permute(1, 0))
                        photon_n_total.append(si.n.torch().to(self.device).permute(1, 0))
                        photon_flux_total.append(photon_flux * throughput.torch().to(self.device).permute(1, 0))
                        photon_rough_total.append(roughness.torch().to(self.device).unsqueeze(0).permute(1, 0))
                        photon_is_glossy_total.append(is_glossy.torch().to(self.device).unsqueeze(0).permute(1, 0))
                        masks.append((torch.norm(photon_pos_total[-1], dim=1) > 0) & (torch.norm(photon_flux_total[-1], dim=1) > 0) & (torch.from_numpy(not_on_delta.numpy()).to(self.device).bool()))
                        
                    throughput *= weight
                    ray = si.spawn_ray(wo_world)
                    dr.eval(ray, bsdf_sample, weight, si, bsdf, active, throughput)
                
                # Concat
                if (MAX_DEPTH > 2):
                    photon_pos_tensor = torch.cat(tuple(x[mask] for x, mask in zip(photon_pos_total, masks)), dim=0)
                    photon_wo_tensor = torch.cat(tuple(x[mask] for x, mask in zip(photon_wo_total, masks)), dim=0)
                    photon_n_tensor = torch.cat(tuple(x[mask] for x, mask in zip(photon_n_total, masks)), dim=0)
                    photon_flux_tensor = torch.cat(tuple(x[mask] for x, mask in zip(photon_flux_total, masks)), dim=0)
                    photon_rough_tensor = torch.cat(tuple(x[mask] for x, mask in zip(photon_rough_total, masks)), dim=0)
                    photon_is_glossy_tensor = torch.cat(tuple(x[mask] for x, mask in zip(photon_is_glossy_total, masks)), dim=0)
                else:
                    photon_pos_tensor = photon_pos_total[0][masks[0]]
                    photon_wo_tensor = photon_wo_total[0][masks[0]]
                    photon_n_tensor = photon_n_total[0][masks[0]]
                    photon_flux_tensor = photon_flux_total[0][masks[0]]
                    photon_rough_tensor = photon_rough_total[0][masks[0]]
                    photon_is_glossy_tensor = photon_is_glossy_total[0][masks[0]]
                photon_map = torch.cat((photon_pos_tensor, photon_wo_tensor, photon_n_tensor, 
                                        photon_rough_tensor, photon_is_glossy_tensor, photon_flux_tensor), dim=1)
                return photon_map

    def eye_pass(self, scene: mi.Scene, sampler: mi.PCG32) -> torch.Tensor:
        dr.make_opaque(sampler)
        with dr.suspend_grad():
            with torch.no_grad():
                active = mi.Bool(True)
                throughput = mi.Color3f(1.0)
                camera: mi.ProjectiveCamera = scene.sensors()[0]
                resolution: mi.Vector2u = camera.film().size()
                cam_width, cam_height = 1.0, 1.0
                pixel_size_x = cam_width / resolution.x
                pixel_size_y = cam_height / resolution.y
                xx = dr.linspace(mi.Float, pixel_size_x / 2, cam_width - pixel_size_x / 2, resolution.x)
                yy = dr.linspace(mi.Float, pixel_size_y / 2, cam_height - pixel_size_y / 2, resolution.y)
                s_x = mi.PCG32(size=resolution.x, initstate=np.random.randint(12352345) if self.stochastic else 42)
                s_y = mi.PCG32(size=resolution.y, initstate=np.random.randint(225325423) if self.stochastic else 7)
                d_x = mi.PCG32(size=resolution.y * resolution.x, initstate=np.random.randint(345345634) if self.stochastic else 24)
                d_y = mi.PCG32(size=resolution.y * resolution.x, initstate=np.random.randint(34563) if self.stochastic else 2025)
                dr.make_opaque(s_x, s_y)
                dr.make_opaque(d_x, d_y)
                jitter_x = (s_x.next_float32() - 0.5) * pixel_size_x
                jitter_y = (s_y.next_float32() - 0.5) * pixel_size_y
                x, y = dr.meshgrid(
                    xx + jitter_x,
                    yy + jitter_y
                )
                ray, _ = camera.sample_ray(0.0, sampler.next_float32(),
                                        mi.Point2f(x, y),
                                        mi.Point2f(d_x.next_float32(), d_y.next_float32()),
                                        active=active)
                
                dr.make_opaque(ray)
                si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)
                bsdf: mi.BSDFPtr = si.bsdf(ray)
                ctx = mi.BSDFContext(mi.TransportMode.Importance)
                wi = mi.Vector3f(si.wi)
                roughness = bsdf.eval_attribute("alpha", si).x
                eta = bsdf.eval_attribute("eta", si)
                k = bsdf.eval_attribute("k", si)
                roughness = dr.select(si.is_valid() & (roughness < 1000), roughness, 0.0)
                is_glossy = si.is_valid() & ((bsdf.flags() & int(mi.BSDFFlags.Glossy)) != 0) & (roughness > 0.1)
                # Check if gather point is on a delta BSDF surface
                active &= si.is_valid() & (((bsdf.flags() & int(mi.BSDFFlags.Delta)) != 0) | ((((bsdf.flags() & int(mi.BSDFFlags.Glossy)) != 0)) & (roughness < 0.1)))

                sampler, si, ray, bsdf, throughput, roughness, is_glossy, active, wi = loop_eye(scene, sampler, si, ray, wi, bsdf, throughput, roughness, is_glossy, active, ctx)
                
                pos = si.p.torch().to(self.device).permute(1, 0)
                norm = si.n.torch().to(self.device).permute(1, 0)
                wi_tensor = wi.torch().to(self.device).permute(1, 0)
                through = throughput.torch().to(self.device).permute(1, 0)
                si.wi = mi.Vector3f(0.0, 0.0, 1.0)
                rough = roughness.torch().to(self.device).unsqueeze(0).permute(1, 0)
                eta = eta.torch().to(self.device).permute(1, 0)
                k = k.torch().to(self.device).permute(1, 0)
                is_glossy_tensor = is_glossy.torch().to(self.device).unsqueeze(0).permute(1, 0)
                bs = bsdf.eval_diffuse_reflectance(si).torch().to(self.device).permute(1, 0)
                return torch.cat((pos, norm, wi_tensor, rough, is_glossy_tensor, eta, k, through * bs), dim=1)
    
    def initialize_radius(self, scene: mi.Scene, sampler: mi.PCG32, scale: float=3.0) -> torch.Tensor:
        dr.make_opaque(sampler)
        with dr.suspend_grad():
            with torch.no_grad():
                active = mi.Bool(True)
                throughput = mi.Color3f(1.0)
                camera: mi.ProjectiveCamera = scene.sensors()[0]
                resolution: mi.Vector2u = camera.film().size()
                cam_width, cam_height = 1.0, 1.0
                pixel_size_x = cam_width / resolution.x
                pixel_size_y = cam_height /resolution.y
                xx = dr.linspace(mi.Float, pixel_size_x / 2, cam_width - pixel_size_x / 2, resolution.x)
                yy = dr.linspace(mi.Float, pixel_size_y / 2, cam_height - pixel_size_y / 2, resolution.y)
                s_x = mi.PCG32(size=resolution.x, initstate=np.random.randint(12352345) if self.stochastic else 42)
                s_y = mi.PCG32(size=resolution.y, initstate=np.random.randint(225325423) if self.stochastic else 7)
                dr.make_opaque(s_x, s_y)
                jitter_x = (s_x.next_float32() - 0.5) * pixel_size_x
                jitter_y = (s_y.next_float32() - 0.5) * pixel_size_y
                x, y = dr.meshgrid(
                    xx + jitter_x,
                    yy + jitter_y
                )

                ray_d, _ = camera.sample_ray_differential(0.0, sampler.next_float32(),
                                        mi.Point2f(x, y),
                                        mi.Point2f(0.0, 0.0),
                                        active=active)
                ray_d.scale_differential(mi.Float(scale))
                origin_ray, dir_ray = ray_d.o, ray_d.d
                origin_x, origin_y = ray_d.o_x, ray_d.o_y
                dir_x, dir_y = ray_d.d_x, ray_d.d_y
                ray: mi.Ray3f = mi.Ray3f(origin_ray, dir_ray)
                dr.make_opaque(ray)
                si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)
                t = dr.norm(si.p - origin_ray)
                bsdf: mi.BSDF = si.bsdf(ray)
                
                # Check if gather point is on a delta BSDF surface (97 is the flag for delta BSDF) 
                active &= si.is_valid() & (((bsdf.flags() & int(mi.BSDFFlags.Delta)) != 0) | ((bsdf.flags() & int(mi.BSDFFlags.Glossy)) != 0))
                ctx = mi.BSDFContext(mi.TransportMode.Importance)

                sampler, si, bsdf, throughput, active, wi_world, total_dist = loop_init(scene, sampler, si, ray, bsdf, throughput, active, ctx, t)
                pos_proj = origin_ray + total_dist * dir_ray
                r_x = origin_x + total_dist * dir_x
                r_y = origin_y + total_dist * dir_y
                diff_x = torch.norm((r_x - pos_proj).torch().permute(1, 0), dim=1)
                diff_y = torch.norm((r_y - pos_proj).torch().permute(1, 0), dim=1)
                radius = torch.clip(torch.maximum(diff_x, diff_y), 0.0, scene.bbox().bounding_sphere().radius / 10)
                return radius