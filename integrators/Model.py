from typing import Tuple
import torch
import torch.nn as nn
import sys
import math
try:
	import tinycudann as tcnn
except ImportError:
	print("This sample requires the tiny-cuda-nn extension for PyTorch.")
	print("You can install it by running:")
	print("============================================================")
	print("tiny-cuda-nn$ cd tools/bindings/torch")
	print("tools/tiny-cuda-nn/bindings/torch$ python setup.py install")
	print("============================================================")
	sys.exit()

from integrators.Utils import compute_gaussian_weighting, init_weights
from integrators.Lookup import GaussianIntegral
import integrators.Activations as act

# For debugging
SYNC = False
EPSILON = 1e-7

class DPMBaseNoEncoder(nn.Module):
    def __init__(self, neighbors: int, dcv_size: int, inchannels: int, outchannels: int, dpm: bool, use_mask=True, 
                 n_hidden_pred: int = 3, finalact: str = "none", size: str ="small", 
                 ratio_matches=False, update_proportion=0.1, stop=100):
        """Base class for the DPM model.
        
        This class is used to define the network architecture and the forward pass.

        Args:
            neighbors (int): number of photons as input
            dcv_size (int): size of the DCV
            dcvaddinonal_dim (int): additional dimension for the DCV for the predictor
            inchannels (int): input dimension of the network
            outchannels (int): output dimension of the network
            dpm (bool): if True, use Deep Photon Mapping (DPM) architecture, means that the DCV and the photon encoding is passed to the predictor
            n_hidden_pred (int, optional): number of hidden layers for the predictor. Defaults to 3.
            finalact (str, optional): activation function for the final layer. Defaults to "none".
        """
        super(DPMBaseNoEncoder, self).__init__()
        if size == "large":
            self.factor = 4
        else:
            self.factor = 1
        self.neighbors = neighbors
        self.dcv_size = dcv_size
        self.use_mask = use_mask
        self.ratio_matches = ratio_matches
        self.update_proportion = update_proportion
        self.stop = stop
        self.inchannels = inchannels
        self.technique = "classic"
        
        
        # Embedder
        config_network = {
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": self.factor * dcv_size // 2,
            "n_hidden_layers": 3,
            "use_bias": True,
            "final_activation": "ReLU",
        }
        
        self.photon_embedder = tcnn.Network(
            n_input_dims=inchannels, 
            n_output_dims=self.factor * dcv_size // 2,
            network_config=config_network)

        # Kernel predictor
        config_network2 = {
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": self.factor * dcv_size // 2,
            "n_hidden_layers": n_hidden_pred,
            "use_bias": True,
            "final_activation": finalact,
        }
        
        self.inputdim_pred = self.factor * dcv_size + (1 if ratio_matches else 0)
        if dpm:
            self.inputdim_pred += self.factor * dcv_size // 2
        
        self.kernel_predictor = tcnn.Network(
            n_input_dims=self.inputdim_pred, 
            n_output_dims=outchannels,
            network_config=config_network2)
   
        self.photon_embedder.apply(init_weights)
        self.kernel_predictor.apply(init_weights)

    @torch.autocast(device_type="cuda")
    def embed(self, photons, device, n_match, ratio_matches) -> Tuple[torch.Tensor, torch.Tensor]:
        # Tiny-cuda-nn expect the inputs to be (batch * neighbors, input_dim)
        photons_tnn = photons.reshape(-1, self.inchannels)
        features = self.photon_embedder(photons_tnn)
        features = features.reshape(-1, self.neighbors,  self.factor * self.dcv_size // 2)
        features = features.permute(0, 2, 1)
        if SYNC:
            torch.cuda.synchronize()
            
        if self.use_mask:
            # Creates mask without using for loops
            range_tensor = torch.arange(self.neighbors, device=device, requires_grad=False).expand(features.shape[0], self.neighbors)
            mask = (range_tensor < n_match.unsqueeze(1)).float().detach()
            if self.ratio_matches:
                features = features * mask.unsqueeze(1).repeat(1, self.dcv_size // 2, 1)
        else:
            mask = None
        
        if SYNC:
            torch.cuda.synchronize()
            
        # Prepare the features for the predictor
        max_x = torch.max(features, 2, keepdim=True)[0]
        mean_x = torch.mean(features, 2, keepdim=True)
        if self.use_mask:
            correction = (self.neighbors / torch.clamp(n_match, 1, self.neighbors))
            correction = correction.unsqueeze(1).unsqueeze(2).repeat(1,  self.factor * self.dcv_size // 2, 1)
            mean_x = mean_x * correction
            if self.ratio_matches:
                features = features * mask.unsqueeze(1).repeat(1, self.dcv_size // 2, 1)
                
        if self.ratio_matches:
            dpc = torch.cat((mean_x, max_x, ratio_matches), 1)
        else:
            dpc = torch.cat((mean_x, max_x), 1)
        
        return dpc, features, mask
class PhotonEncoder(torch.nn.Module):
    def __init__(self, neighbors: int, dcv_size: int, inchannels: int, use_mask=True, n_hidden_pred: int=3):
        """Base class for the photon encoder.
        
        This class is used to define the network architecture and the forward pass.

        Args:
            neighbors (int): number of photons as input
            dcv_size (int): size of the DCV
            inchannels (int): input dimension of the network
            n_hidden_pred (int, optional): number of hidden layers for the predictor. Defaults to 3.
        """
        super(PhotonEncoder, self).__init__()
        self.neighbors = neighbors
        self.dcv_size = dcv_size
        self.use_mask = use_mask
        self.inchannels = inchannels
        
        # Embedder
        config_network = {
            "activation": "Tanh",
            "output_activation": "None",
            "n_neurons": dcv_size // 2,
            "n_hidden_layers": n_hidden_pred,
            "use_bias": True,
            "final_activation": "None",
        }
        
        self.photon_embedder = tcnn.Network(
            n_input_dims=inchannels, 
            n_output_dims=dcv_size // 2,
            network_config=config_network)

    @torch.autocast(device_type="cuda")
    def forward(self, photons, device, n_match) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Tiny-cuda-nn expect the inputs to be (batch * neighbors, input_dim)
        photons_tnn = photons.permute(0, 2, 1)
        photons_tnn = photons_tnn.reshape(-1, self.inchannels)
        features = self.photon_embedder(photons_tnn)
        features = features.reshape(-1, self.neighbors, self.dcv_size // 2)
        features = features.permute(0, 2, 1)
            
        if self.use_mask:
            # Creates mask without using for loops
            range_tensor = torch.arange(self.neighbors, device=device, requires_grad=False).expand(features.shape[0], self.neighbors)
            mask = (range_tensor < n_match.unsqueeze(1)).float().detach()
            features = features * mask.unsqueeze(1).repeat(1, self.dcv_size // 2, 1)
        else:
            mask = None

        # Prepare the features for the predictor
        max_x = torch.max(features, 2, keepdim=True)[0]
        mean_x = torch.mean(features, 2, keepdim=True)
        if self.use_mask:
            correction = (self.neighbors / torch.clamp(n_match, 1, self.neighbors))
            correction = correction.unsqueeze(1).unsqueeze(2).repeat(1, self.dcv_size // 2, 1)
            mean_x = mean_x * correction
            features = features * mask.unsqueeze(1).repeat(1, self.dcv_size // 2, 1)
                
        dpc = torch.cat((mean_x, max_x), 1)
        
        return dpc, features, mask

# Class for the base encoding / decoding scheme
class DPMBase(nn.Module):
    def __init__(self, encoder: PhotonEncoder, neighbors: int, dcv_size: int, inchannels: int, outchannels: int, dpm: bool, use_mask=True, 
                 n_hidden_pred: int = 3, finalact: str = "none", size: str ="small", 
                 ratio_matches=False, update_proportion=0.1, stop=100):
        """Base class for the DPM model.
        
        This class is used to define the network architecture and the forward pass.

        Args:
            encoder (nn.Module): encoder network
            neighbors (int): number of photons as input
            dcv_size (int): size of the DCV
            inchannels (int): input dimension of the network (predictor)
            outchannels (int): output dimension of the network (predictor)
            dpm (bool): if True, use Deep Photon Mapping (DPM) architecture, means that the DCV and the photon encoding is passed to the predictor
            n_hidden_pred (int, optional): number of hidden layers for the predictor. Defaults to 3.
            finalact (str, optional): activation function for the final layer. Defaults to "none".
            size (str, optional): size of the network. Defaults to "small".
            ratio_matches (bool, optional): indicates if we use the ratio of matches in the DCV. Defaults to False.
            update_proportion (float, optional): proportion of gather point used to update the grid. Defaults to 0.1.
            stop (int, optional): number of iterations before stopping to update the grid. Defaults to 100.
        """
        super(DPMBase, self).__init__()
        if size == "large":
            self.factor = 4
        else:
            self.factor = 1
        self.neighbors = neighbors
        self.dcv_size = dcv_size
        self.use_mask = use_mask
        self.ratio_matches = ratio_matches
        self.update_proportion = update_proportion
        self.stop = stop
        self.inchannels = inchannels
        self.technique = "classic"
        
        self.photon_embedder = encoder

        # Kernel predictor
        config_network2 = {
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": self.factor * dcv_size // 2,
            "n_hidden_layers": n_hidden_pred,
            "use_bias": True,
            "final_activation": finalact,
        }
        
        self.inputdim_pred = self.factor * dcv_size + (1 if ratio_matches else 0)
        if dpm:
            self.inputdim_pred += self.factor * dcv_size // 2
        
        self.kernel_predictor = tcnn.Network(
            n_input_dims=self.inputdim_pred, 
            n_output_dims=outchannels,
            network_config=config_network2)
   
        self.photon_embedder.apply(init_weights)
        self.kernel_predictor.apply(init_weights)

    @torch.autocast(device_type="cuda")
    def embed(self, photons, device, n_match, ratio_matches) -> Tuple[torch.Tensor, torch.Tensor]:
        # Tiny-cuda-nn expect the inputs to be (batch * neighbors, input_dim)
        dpc, features, mask = self.photon_embedder(photons, device, n_match)
        if SYNC:
            torch.cuda.synchronize()
                
        if self.ratio_matches:
            dpc = torch.cat((dpc, ratio_matches), 1)
        
        return dpc, features, mask
            

class DPMDirect(DPMBase):
    def __init__(self, neighbors: int, dcv_size: int, size: str = "small"):
        """Direct prediction of the kernel using dpm architecture.
        
        This class is used to define the network architecture and the forward pass.

        Args:
            neighbors (int): number of photons as input
            dcv_size (int): size of the DCV
            size (str, optional): size of the network. Defaults to "small".
        """
        super(DPMDirect, self).__init__(neighbors, dcv_size, inchannels=4, outchannels=1, dpm=True, n_hidden_pred=3, finalact="ReLU", size=size, ratio_matches=False)

    @torch.autocast(device_type="cuda")
    def forward(self, photons, n_match, ratio_matches, radius, dpc_mean, ite, device) -> Tuple[torch.Tensor, torch.Tensor]:
        dpc, features, mask = self.embed(photons, device, n_match, ratio_matches)
        
        # DCV average potentially
        dpc = ((dpc_mean * (ite - 1)) + dpc) / ite
        dpc_repeat = dpc.repeat(1, 1, self.neighbors)
        
        # dpm architecture
        x = torch.cat((dpc_repeat, features), 1)
        
        # Weight predictor
        # (gps, in, photons)
        x_tnn = x.permute(0, 2, 1) 
        # (batch * neighbors, input_dim)
        x_tnn = x_tnn.reshape(-1, self.channels_predictor[0][0])
        result = self.kernel_predictor(x_tnn)
        # (batch * neighbors, output_dim)
        result = result.reshape(-1, self.neighbors, self.channels_predictor[-1][1])
        # (batch, neighbors, output_dim)
        result = result.permute(0, 2, 1)
        # (batch, output_dim, neighbors)
        
        if self.use_mask:
            result = (result[:, 0, :] * mask).unsqueeze(1)

        assert False, "DPMDirect is not implemented yet"

class DPMGaussianAniso(DPMBase):
    def __init__(self, encoder: PhotonEncoder, neighbors: int, dcv_size: int, size: str = "small", ratio_matches=False):
        """Gaussian prediction of the kernel using dpm architecture.
        
        This class is used to define the network architecture and the forward pass.

        Args:
            encoder (nn.Module): encoder network
            neighbors (int): number of photons as input
            dcv_size (int): size of the DCV
            size (str, optional): size of the network. Defaults to "small".
            ratio_matches (bool, optional): indicates if we use the ratio of matches in the DCV. Defaults to False.
        """
        super(DPMGaussianAniso, self).__init__(encoder, neighbors, dcv_size, inchannels=4, outchannels=3, dpm=False, 
                                               n_hidden_pred=3, finalact="None", size=size, ratio_matches=ratio_matches)

        self.act_angle = act.NullAct()
        self.act_scale = act.EluAct()
        self.gaussian_integral = GaussianIntegral()
    
    @torch.autocast(device_type="cuda")
    def forward(self, photons, n_match, ratio_matches, radius, dpc_mean, ite, device) -> Tuple[torch.Tensor, torch.Tensor]:
        dpc, _, mask = self.embed(photons, device, n_match, ratio_matches)
        
        # DCV average potentially
        dpc = ((dpc_mean * (ite - 1)) + dpc) / ite
     
        dpc_permute = dpc.permute(0, 2, 1)
        kernel = self.kernel_predictor(dpc_permute.view(-1, self.inputdim_pred))
        
        theta = self.act_angle(kernel[:, 0]) * 2 * torch.pi
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        scale_x = self.act_scale(kernel[:, 1])
        scale_y = self.act_scale(kernel[:, 2])
        scale_x = torch.clamp(scale_x, torch.tensor([0.00001], device=device), torch.tensor([math.sqrt(3.0)], device=device))
        scale_y = torch.clamp(scale_y, torch.tensor([0.00001], device=device), torch.tensor([math.sqrt(3.0)], device=device))
        
        R = torch.zeros((dpc.shape[0], 2, 2), device=device)
        R[:, 0, 0] = cos_theta
        R[:, 0, 1] = -sin_theta
        R[:, 1, 0] = sin_theta
        R[:, 1, 1] = cos_theta
        
        S = torch.zeros((dpc.shape[0], 2, 2), device=device)
        S[:, 0, 0] = scale_x 
        S[:, 1, 1] = scale_y
        R_T = R.mT
        S_T = S.mT
        covariance = torch.bmm(R, torch.bmm(S, torch.bmm(S_T, R_T)))
        covariance[:, 0, 0] += EPSILON
        covariance[:, 1, 1] += EPSILON
        
        # Compute the photons weights
        weights, cova = compute_gaussian_weighting(photons, covariance, self.neighbors, device)
   
        # Compute the normalization with the tabluation
        normalization = self.gaussian_integral.get(covariance) * radius * radius
        normalization = normalization.unsqueeze(1).repeat(1, self.neighbors)
    
        weights /= (normalization + 0.0000001)
            
        if self.use_mask:
            weights = weights * mask

        if SYNC:
            torch.cuda.synchronize()
    
        return (weights, cova, dpc, R)
    

class DPMGaussianAnisoOpt(DPMBase):
    def __init__(self, encoder: PhotonEncoder, neighbors: int, dcv_size: int, size: str = "small", ratio_matches=False, update_proportion=0.1, stop=100):
        """Gaussian prediction of the kernel using dpm architecture.
        
        This class is used to define the network architecture and the forward pass.

        Args:
            encoder (nn.Module): encoder network
            neighbors (int): number of photons as input
            dcv_size (int): size of the DCV
            size (str, optional): size of the network. Defaults to "small".
            ratio_matches (bool, optional): indicates if we use the ratio of matches in the DCV. Defaults to False.
            update_proportion (float, optional): proportion of gather point used to update the grid. Defaults to 0.1.
            stop (int, optional): number of iterations before stopping to update the grid. Defaults to 100.
        """
        super(DPMGaussianAnisoOpt, self).__init__(encoder, neighbors, dcv_size, inchannels=4, outchannels=3, dpm=False, 
                                                  n_hidden_pred=3, finalact="None", size=size, 
                                                  ratio_matches=ratio_matches, update_proportion=update_proportion, stop=stop)

        self.act_angle = act.NullAct()
        self.act_scale = act.EluAct()
        self.gaussian_integral = GaussianIntegral()
        
    @torch.autocast(device_type="cuda")
    def forward(self, gps, photons, n_match, ratio_matches, grid, ite, device) -> Tuple[torch.Tensor, torch.Tensor]:
        dpc, _, mask = self.embed(photons, device, n_match, ratio_matches)
     
        dpc_permute = dpc.permute(0, 2, 1)
        kernel = self.kernel_predictor(dpc_permute.view(-1, self.inputdim_pred))
        
        theta = self.act_angle(kernel[:, 0]) * 2 * torch.pi
        scales = self.act_scale(kernel[:, 1:])
        return theta, scales

class DPMGaussianAnisoOptGrid(DPMBase):
    def __init__(self, encoder: PhotonEncoder, neighbors: int, dcv_size: int, size: str = "small", ratio_matches=False, stop=100):
        """Gaussian prediction of the kernel using dpm architecture.
        
        This class is used to define the network architecture and the forward pass.

        Args:
            encoder (nn.Module): encoder network
            neighbors (int): number of photons as input
            dcv_size (int): size of the DCV
            size (str, optional): size of the network. Defaults to "small".
            ratio_matches (bool, optional): indicates if we use the ratio of matches in the DCV. Defaults to False.
            stop (int, optional): number of iterations before stopping to update the grid. Defaults to 100.
        """
        super(DPMGaussianAnisoOptGrid, self).__init__(encoder, neighbors, dcv_size, inchannels=4, outchannels=3, dpm=False, 
                                                      n_hidden_pred=3, finalact="None", size=size, ratio_matches=ratio_matches)

        self.act_angle = act.NullAct()
        self.act_scale = act.EluAct()
        self.gaussian_integral = GaussianIntegral()
        self.stop = stop
        
    @torch.autocast(device_type="cuda")
    def forward(self, gps, gps_sub, photons, n_match, ratio_matches, grid, ite, device) -> Tuple[torch.Tensor, torch.Tensor]:
        if ite < self.stop:
            dpc, _, mask = self.embed(photons, device, n_match, ratio_matches)
            grid.update_nearest_cell(gps_sub[:, :3], dpc.squeeze(-1))
        dpc = grid.get_cell_value(gps[:, :3]).unsqueeze(-1)
        dpc_permute = dpc.permute(0, 2, 1)
        kernel = self.kernel_predictor(dpc_permute.view(-1, self.inputdim_pred))
        
        theta = self.act_angle(kernel[:, 0]) * 2 * torch.pi
        scales = self.act_scale(kernel[:, 1:])
        return theta, scales
    
class DPMGaussianAnisoOptGridNoEncoder(DPMBaseNoEncoder):
    def __init__(self, neighbors: int, dcv_size: int, size: str = "small", ratio_matches=False, stop=100):
        """Gaussian prediction of the kernel using dpm architecture.
        
        This class is used to define the network architecture and the forward pass.

        Args:
            neighbors (int): number of photons as input
            dcv_size (int): size of the DCV
            size (str, optional): size of the network. Defaults to "small".
            ratio_matches (bool, optional): indicates if we use the ratio of matches in the DCV. Defaults to False.
            stop (int, optional): number of iterations before stopping to update the grid. Defaults to 100.
        """
        super(DPMGaussianAnisoOptGridNoEncoder, self).__init__(neighbors, dcv_size, inchannels=2, outchannels=3, dpm=False, 
                                                               n_hidden_pred=3, finalact="None", size=size, ratio_matches=ratio_matches)

        self.act_angle = act.NullAct()
        self.act_scale = act.EluAct()
        self.gaussian_integral = GaussianIntegral()
        self.stop = stop
        
    @torch.autocast(device_type="cuda")
    def forward(self, gps, gps_sub, photons, n_match, ratio_matches, grid, ite, device) -> Tuple[torch.Tensor, torch.Tensor]:
        if ite < self.stop:
            dpc, _, mask = self.embed(photons, device, n_match, ratio_matches)
            grid.update_nearest_cell(gps_sub[:, :3], dpc.squeeze(-1))
        dpc = grid.get_cell_value(gps[:, :3]).unsqueeze(-1)
        dpc_permute = dpc.permute(0, 2, 1)
        kernel = self.kernel_predictor(dpc_permute.view(-1, self.inputdim_pred))
        
        theta = self.act_angle(kernel[:, 0]) * 2 * torch.pi
        scales = self.act_scale(kernel[:, 1:])
        return theta, scales

class DPMGaussianIsoOptGrid(DPMBase):
    def __init__(self, neighbors: int, dcv_size: int, size: str = "small", ratio_matches=False, stop=100):
        """Gaussian prediction of the kernel using dpm architecture.
        
        This class is used to define the network architecture and the forward pass.

        Args:
            neighbors (int): number of photons as input
            dcv_size (int): size of the DCV
            size (str, optional): size of the network. Defaults to "small".
            ratio_matches (bool, optional): indicates if we use the ratio of matches in the DCV. Defaults to False.
            stop (int, optional): number of iterations before stopping to update the grid. Defaults to 100.
        """
        super(DPMGaussianIsoOptGrid, self).__init__(neighbors, dcv_size, inchannels=4, outchannels=1, dpm=False, 
                                                    n_hidden_pred=3, finalact="None", size=size, ratio_matches=ratio_matches)

        self.act_angle = act.NullAct()
        self.act_scale = act.EluAct()
        self.gaussian_integral = GaussianIntegral()
        self.stop = stop
        
    @torch.autocast(device_type="cuda")
    def forward(self, gps, gps_sub, photons, n_match, ratio_matches, grid, ite, device) -> Tuple[torch.Tensor, torch.Tensor]:
        if ite < self.stop:
            dpc, _, mask = self.embed(photons, device, n_match, ratio_matches)
            grid.update_nearest_cell(gps_sub[:, :3], dpc.squeeze(-1))
        dpc = grid.get_cell_value(gps[:, :3]).unsqueeze(-1)
        dpc_permute = dpc.permute(0, 2, 1)
        kernel = self.kernel_predictor(dpc_permute.view(-1, self.inputdim_pred))
        
        theta = torch.zeros(dpc_permute.shape[0], device=device, dtype=kernel.dtype)
        scale = self.act_scale(kernel[:, 0])
        scales = torch.stack((scale, scale), dim=1)
        return theta, scales

class DPMGaussianAnisoOptDecoder(DPMBase):
    def __init__(self, neighbors: int, dcv_size: int, size: str = "small", ratio_matches=False):
        """Gaussian prediction of the kernel using dpm architecture.
        
        This class is used to define the network architecture and the forward pass.

        Args:
            neighbors (int): number of photons as input
            dcv_size (int): size of the DCV
            size (str, optional): size of the network. Defaults to "small".
            ratio_matches (bool, optional): indicates if we use the ratio of matches in the DCV. Defaults to False.
        """
        super(DPMGaussianAnisoOptDecoder, self).__init__(neighbors, dcv_size, inchannels=4, outchannels=3, dpm=False, 
                                                         n_hidden_pred=3, finalact="None", size=size, ratio_matches=ratio_matches)

        self.act_angle = act.NullAct()
        self.act_scale = act.EluAct()
        self.gaussian_integral = GaussianIntegral()
        
    @torch.autocast(device_type="cuda")
    def forward(self, dpc_permute) -> Tuple[torch.Tensor, torch.Tensor]:
        kernel = self.kernel_predictor(dpc_permute.view(-1, self.inputdim_pred))
        
        theta = self.act_angle(kernel[:, 0]) * 2 * torch.pi
        scales = self.act_scale(kernel[:, 1:])
        return theta, scales

class DPMGaussianAnisoOptEncoder(DPMBase):
    def __init__(self, neighbors: int, dcv_size: int, size: str = "small", ratio_matches=False):
        """Gaussian prediction of the kernel using dpm architecture.
        
        This class is used to define the network architecture and the forward pass.

        Args:
            neighbors (int): number of photons as input
            dcv_size (int): size of the DCV
            size (str, optional): size of the network. Defaults to "small".
            ratio_matches (bool, optional): indicates if we use the ratio of matches in the DCV. Defaults to False.
        """
        super(DPMGaussianAnisoOptEncoder, self).__init__(neighbors, dcv_size, inchannels=4, outchannels=3, dpm=False, 
                                                         n_hidden_pred=3, finalact="None", size=size, ratio_matches=ratio_matches)

        self.act_angle = act.NullAct()
        self.act_scale = act.EluAct()
        self.gaussian_integral = GaussianIntegral()
        
    @torch.autocast(device_type="cuda")
    def forward(self, photons, n_match, ratio_matches, radius, dpc_mean, ite, device) -> Tuple[torch.Tensor, torch.Tensor]:
        dpc, _, mask = self.embed(photons, device, n_match, ratio_matches)
     
        dpc_permute = dpc.permute(0, 2, 1)
        return dpc_permute

class DPMGaussianAnisoOptGridEncoder(DPMBase):
    def __init__(self, neighbors: int, dcv_size: int, size: str = "small", ratio_matches=False, update_proportion=0.1, stop=100):
        """Gaussian prediction of the kernel using dpm architecture.
        
        This class is used to define the network architecture and the forward pass.

        Args:
            neighbors (int): number of photons as input
            dcv_size (int): size of the DCV
            size (str, optional): size of the network. Defaults to "small".
            ratio_matches (bool, optional): indicates if we use the ratio of matches in the DCV. Defaults to False.
            update_proportion (float, optional): proportion of gather point used to update the grid. Defaults to 0.1.
            stop (int, optional): number of iterations before stopping to update the grid. Defaults to 100.
        """
        super(DPMGaussianAnisoOptGridEncoder, self).__init__(neighbors, dcv_size, inchannels=4, outchannels=3, dpm=False, 
                                                             n_hidden_pred=3, finalact="None", size=size, ratio_matches=ratio_matches)

        self.act_angle = act.NullAct()
        self.act_scale = act.EluAct()
        self.gaussian_integral = GaussianIntegral()
        self.update_proportion = update_proportion
        self.stop = stop
        
    @torch.autocast(device_type="cuda")
    def forward(self, gps, gps_sampled, photons, n_match, ratio_matches, grid, ite, device) -> Tuple[torch.Tensor, torch.Tensor]:
        if ite < self.stop:
            dpc, _, mask = self.embed(photons, device, n_match, ratio_matches)
            grid.update_nearest_cell(gps_sampled[:, :3], dpc.squeeze(-1))
        dpc = grid.get_cell_value(gps[:, :3]).unsqueeze(-1)
        dpc_permute = dpc.permute(0, 2, 1)
        return dpc_permute

