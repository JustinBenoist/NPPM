import os
import gc
import math
from typing import Tuple, List
from tqdm import tqdm
import numpy as np
import torch
from torch.distributions import Distribution
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
import mitsuba as mi
import drjit as dr
import pyexr

from integrators.Utils import radius_search
from integrators.NPPM import PPMIntegrator
from integrators.Model import DPMBase


def estimate_density(cum_flux: torch.Tensor, radius: float, total_photons: int) -> torch.Tensor:
    # WARNING : the second pi is to account for diffuse brdf on the gather point
    return cum_flux / (torch.pi * radius * radius * total_photons)

def tone_mapping(flux: torch.Tensor) -> torch.Tensor:
    return flux / (flux + 10.0)

def tone_mapping_2(flux):
    a = 0.01
    log_flux_a = torch.log(flux + a)
    log_a = math.log(a)
    num = log_flux_a - log_a
    denum = log_flux_a - log_a + 1
    return num / denum

class DPMTrainer:
    def __init__(self, network: DPMBase, optimizer: Optimizer, scheduler : LRScheduler, 
                 distribution : Distribution, radius: torch.Tensor, batch_size: int, device: str, 
                 loss_function=torch.nn.MSELoss(), photons_per_iter: int=300, random_density: bool=False,
                 pure_caustic: bool=True, ratio_matches: bool=False, knn_mode: bool=False, gaussian: bool=True,
                 freeze_encoder: bool=True) -> None:
        self.network = network
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.distribution = distribution
        self.radius = radius
        self.batch_size = batch_size
        self.device = device
        self.loss_function = loss_function
        self.photons_per_iter = photons_per_iter
        self.ppm = PPMIntegrator(photons_per_iter, 1, device=device,
                                init_radius=radius, pure_caustic=pure_caustic)
        self.random_density = random_density
        self.ratio_matches = ratio_matches
        self.knn_mode = knn_mode
        self.deterministic = False
        self.gaussian = gaussian
        self.freeze_encoder = freeze_encoder
        
    def eval_dataset(self, scenes: List[mi.Scene], 
                     ground_truth: torch.Tensor, mask: torch.Tensor,
                     training: bool, seed: int,
                     pbar: tqdm) -> Tuple[List[float], List[float], torch.Tensor, torch.Tensor, int]:
        # np.random.seed(0)
        # torch.manual_seed(0)
        
        # Compute some sizes for batches
        if training:
            scene = scenes[0]
        else:
            scene = scenes
        film: mi.Film = scene.sensors()[0].film()
        N = film.size().x * film.size().y
        radius_data = torch.ones(N).to(self.device) * self.radius
        flux_correc_data = torch.ones((N, 1)).to(self.device)
        if training:
            self.network.train()
            size_test = 0
            nb_batch = 200
        else:
            self.network.eval()
            size_test = N
            nb_batch = size_test // self.batch_size
        if self.freeze_encoder:
            for param in self.network.photon_embedder.parameters():
                param.requires_grad = False
            self.network.photon_embedder.eval()
        tab_loss, box_loss = [], []

        res_image = torch.zeros((N, 3), device=self.device)
        box_image = torch.zeros((N, 3), device=self.device)
        # Dataset preparation
        selec_size = N if training else N
        if training:
            indices = np.random.permutation(N)
        else:
            indices = np.arange(N - size_test, N, 1)

        neighbors = self.network.neighbors
        
        # Shooting photons and gather points for training
        sampler_eye = mi.PCG32(size=N, initstate=0 if self.deterministic else seed)
        seed += 1
        sampler = mi.PCG32(size=self.photons_per_iter, initstate=0 if self.deterministic else seed)
        seed += 1

        gps = [self.ppm.eye_pass(scene, sampler_eye).detach().cpu()]
        gts = [ground_truth[0, :, :]]
        photon_maps = []
        photon_maps.append(self.ppm.photon_pass(scene, sampler, 0).cpu())
        if training:
            for i in range(1, len(scenes)):
                scene = scenes[i]
                sampler_eye = mi.PCG32(size=N, initstate=0 if self.deterministic else seed) # Must be here or everything breaks
                seed += 1
                sampler = mi.PCG32(size=self.photons_per_iter, initstate=0 if self.deterministic else seed)
                seed += 1

                gps.append(self.ppm.eye_pass(scene, sampler_eye).detach().cpu())
                gts.append(ground_truth[i, :, :])
                photon_maps.append(self.ppm.photon_pass(scene, sampler, 0).cpu())
            
        # Training
        for idx_scene in range(len(scenes) if training else 1):
            loss_avg_scene = 0
            idx_scene = idx_scene if training else -1
            sampler_eye = mi.PCG32(size=N, initstate=0 if self.deterministic else seed)
            if self.random_density and training:
                random_radius = 4.0 + 0.5 * (2 * torch.rand_like(radius_data, device=self.device) - 1)
                radius_data = self.ppm.initialize_radius(scenes[idx_scene], sampler_eye, random_radius)
            if not training:
                radius_data = self.ppm.initialize_radius(scene, sampler_eye, 3)
            if self.knn_mode:
                raise NotImplementedError("KNN mode is not implemented yet.")
            else:
                photons_data, flux_data, flux_correc_data, n_matches_data, _ = radius_search(photon_maps[idx_scene].to(self.device), gps[idx_scene].to(self.device),
                                                                                             radius_data, self.network.neighbors, self.device)
            
            gt_data = gts[idx_scene]
            photons_data = photons_data.detach()
            # wo_data = wo_data.detach()
            flux_data = flux_data.detach()
            flux_correc_data = flux_correc_data.detach()
            n_matches_data = n_matches_data.detach()
            gt_data = gt_data.detach().to(self.device)
            mask_data = mask[idx_scene].to(self.device)
            dr.eval()
            
            for i in range(nb_batch):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                index_range = slice(start, end)
                photons = torch.nan_to_num(photons_data[indices[index_range], :, :], 0.0).to(self.device)
                photon_flux = torch.nan_to_num(flux_data[indices[index_range], :], 0.0)
                flux_correc = flux_correc_data[indices[index_range]]
                n_match = n_matches_data[indices[index_range]]
                radius = radius_data[indices[index_range]]

                # Position input
                inputs = photons[:, :2, :] / radius.unsqueeze(1).unsqueeze(2)
                # Flux input
                ratio_matches = torch.log(1 + (100 / (n_match + 1))).unsqueeze(1).unsqueeze(2)
                ratio_matches = torch.clamp(ratio_matches, torch.tensor([0.0], device=self.device), torch.tensor([4.0], device=self.device))
                network_input = torch.nan_to_num(inputs, 0)
                f_size = self.network.dcv_size + (1 if self.ratio_matches else 0)
                dpc = torch.zeros((self.batch_size, f_size), device=self.device).detach()
                if self.network.technique == "classic":
                    dpc = dpc.unsqueeze(2)

                # DPM computes the kernel
                if self.gaussian:
                    inference = self.network(network_input, n_match, ratio_matches, radius, dpc, 1, self.device)
                    pred_kernel, cov, _, R = inference
                    estimate = torch.sum(pred_kernel.unsqueeze(1) * photon_flux, dim=2) * flux_correc / self.photons_per_iter
                else:
                    pred_kernel, _ = self.network(network_input, n_match, ratio_matches, radius, dpc, 1, self.device)
                    pred_kernel = pred_kernel.view((photons.shape[0], neighbors))
                    estimate = torch.sum(pred_kernel.unsqueeze(1) * photon_flux, dim=2) * flux_correc / (self.photons_per_iter * radius.unsqueeze(1) * radius.unsqueeze(1))
                box = torch.sum(photon_flux, dim=2) * flux_correc * (1 - mask_data[indices[index_range]])
                box = estimate_density(box, radius.unsqueeze(1), self.photons_per_iter)
                
                estimate = estimate * (1 - mask_data[indices[index_range]])
                gt = gt_data[indices[index_range]] * (1 - mask_data[indices[index_range]])
                
                loss = self.loss_function(estimate, gt)

                box_l = self.loss_function(box, gt)
                if training:
                    loss.backward()
                    self.optimizer.step()
                else:
                    res_image[indices[index_range], :] = estimate.detach()
                    box_image[indices[index_range], :] = box.detach()
                tab_loss.append(loss.item())
                loss_avg_scene += loss.item()
                box_loss.append(box_l.item())
                self.optimizer.zero_grad(set_to_none=True)
                dr.eval()

            # Give more feedback on the training
            loss_avg_scene /= nb_batch
            if training:
                pbar.set_description(f"Training {idx_scene + 1} / {len(scenes)}, training loss : {loss_avg_scene:.4f}")
            else:
                pbar.set_description(f"Testing, testing loss : {loss_avg_scene:.4f}")
        gc.collect()
        torch.cuda.empty_cache()

        if training:
            self.scheduler.step()
        return tab_loss, box_loss, res_image, box_image, seed

    def run(self, scenes: List[mi.Scene], scene_test: mi.Scene, 
            ground_truth: torch.Tensor, ground_truth_test: torch.Tensor, 
            mask: torch.Tensor, mask_test: torch.Tensor,
            epochs: int,
            output: str) -> Tuple[List[float], List[float], List[float]]:
        avg_train_loss = []
        avg_test_loss = []
        avg_box_loss = []
        pbar = tqdm(range(epochs))
        film: mi.Film = scene_test.sensors()[0].film()
        res_x, res_y = film.size().x, film.size().y
        seed = 0
        
        # First test (network not trained yet)
        print("First test")
        with torch.no_grad():
            test_loss, box_loss, pred, box, seed  = self.eval_dataset(scene_test, ground_truth_test, mask_test, training=False, seed=seed, pbar=pbar)
        pyexr.write(os.path.join(output, f"pred_0.exr"), pred.reshape(res_x, res_y, 3).cpu().detach().numpy().astype(np.float32))
        pyexr.write(os.path.join(output, f"box.exr"), box.reshape(res_x, res_y, 3).cpu().detach().numpy().astype(np.float32))
        
        print("Training")
        for epoch in pbar:
            
            train_loss, _, _, _, seed = self.eval_dataset(scenes, ground_truth, mask, training=True, seed=seed, pbar=pbar)
            avg_train_loss.append(np.mean(train_loss))
                
            with torch.no_grad():
                test_loss, box_loss, pred, box, seed  = self.eval_dataset(scene_test, ground_truth_test, mask_test, training=False, seed=0, pbar=pbar)
            pyexr.write(os.path.join(output, f"pred_{epoch + 1}.exr"), pred.reshape(res_x, res_y, 3).cpu().detach().numpy().astype(np.float32))
            torch.save(self.network.state_dict(), os.path.join(output, f"model_{epoch}.pth"))
            avg_test_loss.append(np.mean(test_loss))
            avg_box_loss.append(np.mean(box_loss))
            pbar.set_description(f"Training {epoch + 1} / {epochs}, training loss : {avg_train_loss[-1]:.4f}")
        return avg_train_loss, avg_test_loss, avg_box_loss
