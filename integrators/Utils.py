import numpy as np
import torch
import torch.nn as nn
import frnn
from tinycudann.modules import Network


def compute_gaussian_weighting(photons: torch.Tensor, covariance: torch.Tensor, neighbors: int, device) -> np.ndarray:
    weights = torch.zeros((photons.shape[0], photons.shape[2]), device=device)
    points = photons[:, :3, :]  # Avoid deepcopy for better performance
    proj_2d = points[:, 0:2]
    
    # Precompute reshaped covariance matrix and 2D projections
    covariance_matrix = covariance.view((points.shape[0], 2, 2, 1)).expand(-1, -1, -1, neighbors)
    covariance_matrix = covariance_matrix.permute(0, 3, 1, 2).reshape(-1, 2, 2)
    proj_2d_reshaped = proj_2d.permute(0, 2, 1).reshape(-1, 2)
    
    # Compute Gaussian weights
    weights = anisotropic_gaussian_xy(proj_2d_reshaped, covariance_matrix).view(points.shape[0], points.shape[2])
    return weights, covariance.view((points.shape[0], 2, 2))

def anisotropic_gaussian(r, theta, a, b):
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    det = a * b
    normalization = 1 / (2 * np.pi * torch.sqrt(det))
    # return normalization * torch.exp(-0.5 * ((x / a)**2 + (y / b)**2))
    return normalization * torch.exp(-(0.5 / det) * ((b * x**2) + (a * y**2)))

def anisotropic_gaussian_xy(points, cov):
    a = cov[:, 0, 0]
    b = cov[:, 1, 1]
    c = cov[:, 1, 0]
    x = points[:, 0]
    y = points[:, 1]
    det = a * b - c**2
    normalization = 1 / (2 * np.pi * torch.sqrt(det))
    return normalization * torch.exp(-(0.5 / det) * ((b * x**2) + (a * y**2) - 2 * c * x * y))

def truncated_gaussian_ratio_pdf(points, r):
    x = points[:, 0]
    y = points[:, 1]
    radius2 = torch.square(x) + torch.square(y)
    mask = (radius2 < torch.square(r)).float()
    return mask

def gauss_MC_batch(cov, n_samples, radius, R, neighbors, device):
    n = n_samples
    batch_size = cov.shape[0]
    R_T = torch.swapaxes(R, 1, 2)
    cov_ortho = torch.bmm(R, torch.bmm(cov, R_T))
    a_tensor, b_tensor = cov_ortho[:, 0, 0], cov_ortho[:, 1, 1]
    a_tensor = a_tensor.unsqueeze(1).repeat(1, n).reshape((batch_size * n))
    b_tensor = b_tensor.unsqueeze(1).repeat(1, n).reshape((batch_size * n))
    radii = radius.unsqueeze(1).repeat(1, n).reshape((batch_size * n))
    distrib_uni = torch.distributions.Uniform(low=torch.zeros(n * batch_size, device=device),
                                        high=torch.ones(n * batch_size, device=device))
    r = radii * torch.sqrt(distrib_uni.sample((1,)))
    theta = distrib_uni.sample((1,)) * 2 * np.pi
    values_uni = anisotropic_gaussian(r, theta, a_tensor, b_tensor)
    return torch.mean(values_uni.reshape(batch_size, n), dim=1) * np.pi * radius * radius

def deepcopy_dict(d: dict):
    new_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            new_dict[key] = deepcopy_dict(value)
        else:
            new_dict[key] = value
    return new_dict

def batch_rotation_to_z(vectors):
    batch_size = vectors.shape[0]

    # Normalize input vectors
    vectors = vectors / (vectors.norm(dim=1, keepdim=True) + 1e-8)

    # Target direction (z-axis)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=vectors.device).expand(batch_size, -1)

    # Compute rotation axis (cross product with (0,0,1))
    rotation_axis = torch.cross(vectors, z_axis, dim=1)

    # Compute sine and cosine of the rotation angle
    cos_theta = torch.sum(vectors * z_axis, dim=1)  # dot product
    sin_theta = rotation_axis.norm(dim=1)

    # Normalize rotation axis
    rotation_axis = torch.where(
        sin_theta[:, None] > 1e-8, 
        rotation_axis / (sin_theta[:, None] + 1e-8), 
        torch.zeros_like(rotation_axis)  # Set axis to zero vector if already aligned
    )

    # Construct skew-symmetric matrix for cross product
    K = torch.zeros((batch_size, 3, 3), device=vectors.device)
    K[:, 0, 1] = -rotation_axis[:, 2]
    K[:, 0, 2] = rotation_axis[:, 1]
    K[:, 1, 0] = rotation_axis[:, 2]
    K[:, 1, 2] = -rotation_axis[:, 0]
    K[:, 2, 0] = -rotation_axis[:, 1]
    K[:, 2, 1] = rotation_axis[:, 0]

    # Compute rotation matrix using Rodrigues' formula: R = I + sinθ K + (1 - cosθ) K^2
    I = torch.eye(3, device=vectors.device).expand(batch_size, -1, -1)
    R = I + sin_theta[:, None, None] * K + (1 - cos_theta[:, None, None]) * (K @ K)

    return R

def radius_search_ppm(photons, gps, radius, 
                    # CPPM parameters (None means unused)
                    is_cppm, total_count, min_count, photon_counts):
    points = photons[:, :3]
    wo_photons = photons[:, 3:6]
    normals_photons = photons[:, 6:9]
    roughness_photons = photons[:, 9:10]
    is_glossy_photons = photons[:, 10:11].to(torch.int32)
    flux = photons[:, -3:]
    query = gps[:, :3]
    normals = gps[:, 3:6]
    direction = gps[:, 6:9]
    roughness = gps[:, 9:10]
    is_glossy = gps[:, 10:11].to(torch.int32)
    eta = gps[:, 11:14]
    k = gps[:, 14:17]
    albedo = gps[:, -3:].unsqueeze(2)
    total_c = total_count[None] if is_cppm else None
    min_c = min_count[None] if is_cppm else None
    photon_c = photon_counts[None] if is_cppm else None

    return frnn.frnn_grid_photon_gather(points[None], normals_photons[None], wo_photons[None], roughness_photons[None],
                                        is_glossy_photons[None], flux[None], query[None], normals[None], direction[None],
                                        roughness[None], is_glossy[None], eta[None], k[None], albedo[None], radius[None],
                                        is_cppm=is_cppm, total_count=total_c, min_counts=min_c, photon_counts=photon_c)

def radius_search_transform(photons, gps, radius, max_neighbors, grid=None):
    points = photons[:, :3]
    normals_photons = photons[:, 6:9]
    query = gps[:, :3]
    normals = gps[:, 3:6]
    flux = photons[:, -3:]
    
    input_enc, n_matches, grid = frnn.frnn_grid_resampling_transform(query[None], points[None], 
                                                                    r=radius[None], 
                                                                    p_normal=normals_photons[None], g_normal=normals[None], flux_avg=flux.mean(dim=1, keepdim=True)[None], 
                                                                    K=max_neighbors, grid=grid)
    return input_enc, n_matches, grid

def radius_search(photons, gps, radius, max_neighbors, device, grid=None):
    points = photons[:, :3]
    normals_photons = photons[:, 6:9]
    flux = photons[:, -3:]
    query = gps[:, :3]
    normals = gps[:, 3:6]
    albedo = gps[:, -3:].unsqueeze(2)
    
    # edges_index, n_matches = frnn.frnn_bf_resampling(query.unsqueeze(0), points.unsqueeze(0), r=radius[0], K=max_neighbors)
    edges_index, n_matches, grid = frnn.frnn_grid_resampling(query[None], points[None], r=radius[None], K=max_neighbors, grid=grid)
    edges_index = edges_index.to(torch.int64)
    
    normals_photons_match = frnn.frnn_gather(normals_photons[None], edges_index)[0]
    mask_normals = (torch.einsum("nmd,nmd->nm", normals_photons_match, normals[:, None, :]) < 0.9)
    
    res = frnn.frnn_gather(points[None], edges_index)[0] - query[:, None, :]
    rotations = batch_rotation_to_z(normals)[:, None, :, :]  # [N, 1, 3, 3]

    # Add random Z-axis rotation
    N = rotations.shape[0]
    theta = torch.rand(N, device=rotations.device) * 2 * torch.pi
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    Rz = torch.zeros((N, 3, 3), device=rotations.device)
    Rz[:, 0, 0] = cos_theta
    Rz[:, 0, 1] = -sin_theta
    Rz[:, 1, 0] = sin_theta
    Rz[:, 1, 1] = cos_theta
    Rz[:, 2, 2] = 1.0
    rotations = torch.matmul(Rz[:, None, :, :], rotations)
    
    res = torch.einsum("nmij,nmj->nmi", rotations, res).permute(0, 2, 1)
    
    mask_expanded = mask_normals.unsqueeze(1).expand(-1, 3, -1)
    res.masked_fill_(mask_expanded, 0.0)
    
    flux_res = frnn.frnn_gather(flux[None], edges_index)[0]
    flux_res.masked_fill_(mask_normals.unsqueeze(-1), 0.0)
    flux_res = (flux_res.permute(0, 2, 1) * albedo) / torch.pi
    
    n_matches = n_matches.squeeze()
    flux_correction = torch.ones_like(n_matches, device=device, dtype=torch.float32)
    flux_correction[n_matches > max_neighbors] = n_matches[n_matches > max_neighbors] / max_neighbors
    
    return res, flux_res, flux_correction.unsqueeze(1), n_matches, grid

def init_weights(m: Network):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, Network):
        pass
        # torch.nn.init.constant_(m.params, 0.01)
        # torch.nn.init.constant_(m.bias, 0.0001)
        # torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        # torch.nn.init.xavier_uniform_(m.params, gain=1.0)


if __name__ == "__main__":
    pass