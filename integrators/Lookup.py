import os
import math
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

from integrators.Utils import gauss_MC_batch

class GaussianIntegral:
    def __init__(self):
        # Resolution of the lookup table
        self.RES = 1024
        # Number of samples to build the lookup table
        self.N = self.RES**2
        self.x = torch.logspace(math.log(0.0001), math.log(50.0), self.RES, base=math.exp(1), device="cuda", requires_grad=False)
        self.y = torch.logspace(math.log(0.0001), math.log(50.0), self.RES, base=math.exp(1), device="cuda", requires_grad=False)
        file_path = "integrators/integral_lookup.pt"
        if not os.path.exists(file_path):
            print("Cut gaussian integral values lookup table does not exists building it now ...")
            self.__build_lut(n_samples=1000000)
        # WARN: the lookup table could have different size, this should create issues
        print("Loading cut gaussian integral values lookup table ...")
        self.values = torch.load(file_path, map_location="cuda").reshape(self.RES, self.RES).unsqueeze(0).unsqueeze(1)
        self.x_min = self.x[0]
        self.x_max = self.x[-1]
        self.y_min = self.y[0]
        self.y_max = self.y[-1]
    
    def get(self, cov: torch.Tensor) -> torch.Tensor:
        x, y = cov[:, 0, 0], cov[:, 1, 1]
        res = torch.ones(x.shape[0], device="cuda")
        u = (x - y) * math.sqrt(2) / 2
        v = (x + y) * math.sqrt(2) / 2
        needs_integration = torch.logical_or(v > 0.0424 * torch.square(u) + 0.0868, torch.norm(torch.stack((x, y), dim=-1), dim=1) > 0.09)
        res[needs_integration] = self.__interpolate_integral_batch(x[needs_integration], y[needs_integration])
        return res

    def __interpolate_integral_batch(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        """Interpolates integral values for a batch of (x, y) points."""
        assert x_batch.shape == y_batch.shape, "x_batch and y_batch must have the same shape!"
        
        # Normalize to [-1, 1] for grid_sample
        x_norm = 2 * (torch.log(x_batch) - torch.log(self.x_min)) / (torch.log(self.x_max) - torch.log(self.x_min)) - 1
        y_norm = 2 * (torch.log(y_batch) - torch.log(self.y_min)) / (torch.log(self.y_max) - torch.log(self.y_min)) - 1

        # Stack into grid format (B, 1, 1, 2) for batch processing
        grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(1).unsqueeze(1).permute(1, 2, 0, 3)  # Shape (B, 1, 1, 2)

        # Use bilinear interpolation
        return F.grid_sample(self.values, grid, mode="bilinear", align_corners=True).squeeze()
    
    def __build_lut(self, n_samples: int=1000000):
        with torch.no_grad():
            R = torch.eye(2, device="cuda").unsqueeze(0).repeat(self.N, 1, 1)
            radii = torch.ones(self.N, device="cuda")
            xx, yy = torch.meshgrid(self.x, self.y)
            xy = torch.cat((xx.reshape(self.N).unsqueeze(1), yy.reshape(self.N).unsqueeze(1)), dim=1)
            norms = torch.ones(self.N, device="cuda")
            rot = (math.sqrt(2) / 2) * torch.tensor([[1 , -1], [1, 1]], device="cuda")
            # # print(torch.tensor([]))
            uv = xy@rot.T
            mask = torch.logical_or(uv[:, 1] > 0.0424 * torch.square(uv[:, 0]) + 0.0868, torch.norm(xy, dim=1) > 0.09)
            pbar = tqdm(torch.where(mask)[0])
            
            for i in pbar:
                covs = torch.eye(2, device="cuda").unsqueeze(0)
                covs[0, 0, 0] = xy[i, 0]
                covs[0, 1, 1] = xy[i, 1]
                norms[i] = min(gauss_MC_batch(covs, n_samples, radii[:1], R[:1, :, :], 0, "cuda"), 1.0)
                
            torch.save(norms, "integrators/integral_lookup.pt")
            print(f"Lookup table size = {norms.element_size() * norms.nelement()} bytes")
        
        plt.imshow(norms.reshape(self.RES, self.RES).detach().cpu().numpy())
        plt.show()


if __name__ == "__main__":
    gauss_lut = GaussianIntegral()