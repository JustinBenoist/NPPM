#include "gather/gather_ppm.h"

#include <curand_kernel.h>

__global__ void GatherKernelV1(
    // Gatherpoints
    const float *__restrict__ points1, 
    // Photons
    const float *__restrict__ points2,
    const int64_t *__restrict__ lengths1, const int64_t *__restrict__ lengths2,
    const int *__restrict__ pc2_grid_off,
    const int *__restrict__ sorted_points1_idxs,
    const int *__restrict__ sorted_points2_idxs,
    const float *__restrict__ params,
    const float *__restrict__ p_normal,
    const float *__restrict__ p_dir,
    const float *__restrict__ p_rough,
    const int *__restrict__ p_is_glossy,
    const float *__restrict__ p_flux,
    float *__restrict__ p_visible,
    const float *__restrict__ g_normal,
    const float *__restrict__ g_dir,
    const float *__restrict__ g_rough,
    const int *__restrict__ g_is_glossy,
    const float *__restrict__ g_eta,
    const float *__restrict__ g_k,
    const float *__restrict__ g_albedo,

    float *__restrict__ value,

    int N, int P1, int P2, int G,
    const float *__restrict__ rs) {
  const int D = 3;

  // Position of the point (3d?)
  float cur_point[D];
  float cur_normal[D];
  float cur_dir[D];
  float cur_rough;
  bool cur_is_glossy;
  float cur_eta[D];
  float cur_k[D];
  float cur_albedo[D];
  // Result
  float result[D];

  int chunks_per_cloud = (1 + (P1 - 1) / blockDim.x);
  int chunks_to_do = N * chunks_per_cloud;
  for (int chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    int n = chunk / chunks_per_cloud;
    int start_point = blockDim.x * (chunk % chunks_per_cloud);
    int p1 = start_point + threadIdx.x;
    int old_p1 = sorted_points1_idxs[n * P1 + p1];
    if (p1 >= lengths1[n]) {
      continue;
    }

    // Radius get not sorted from the input (position)
    float cur_r = rs[n * P1 + old_p1];
    float cur_r2 = cur_r * cur_r;
    float kernel = 1.0 / (cur_r2 * M_PI);

    cur_rough = g_rough[n * P1 + old_p1];
    cur_is_glossy = g_is_glossy[n * P1 + old_p1] == 1;

    // Load data
    for (int d = 0; d < D; ++d) {
      // Point get reordered
      cur_point[d] = points1[n * P1 * D + p1 * D + d];
      // Other information are not reordered
      cur_normal[d] = g_normal[n * P1 * D + old_p1 * D + d];
      cur_dir[d] = g_dir[n * P1 * D + old_p1 * D + d];
      cur_albedo[d] = g_albedo[n * P1 * D + old_p1 * D + d];
      cur_eta[d] = g_eta[n * P1 * D + old_p1 * D + d];
      cur_k[d] = g_k[n * P1 * D + old_p1 * D + d];
    }

    // Accumulated flux
    for (int d = 0; d < D; ++d) {
      result[d] = 0;
    }

    float grid_min_x = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_X];
    float grid_min_y = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_Y];
    float grid_min_z = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_Z];
    float grid_delta = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_DELTA];
    int grid_res_x = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_X];
    int grid_res_y = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_Y];
    int grid_res_z = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_Z];
    int grid_total = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_TOTAL];

    int min_gc_x =
        (int)std::floor((cur_point[0] - grid_min_x - cur_r) * grid_delta);
    int min_gc_y =
        (int)std::floor((cur_point[1] - grid_min_y - cur_r) * grid_delta);
    int min_gc_z =
        (int)std::floor((cur_point[2] - grid_min_z - cur_r) * grid_delta);
    int max_gc_x =
        (int)std::floor((cur_point[0] - grid_min_x + cur_r) * grid_delta);
    int max_gc_y =
        (int)std::floor((cur_point[1] - grid_min_y + cur_r) * grid_delta);
    int max_gc_z =
        (int)std::floor((cur_point[2] - grid_min_z + cur_r) * grid_delta);
    
    // Search inside the grid
    for (int x = max(min_gc_x, 0); x <= min(max_gc_x, grid_res_x - 1); ++x) {
      for (int y = max(min_gc_y, 0); y <= min(max_gc_y, grid_res_y - 1); ++y) {
        for (int z = max(min_gc_z, 0); z <= min(max_gc_z, grid_res_z - 1);
             ++z) {
          int cell_idx = (x * grid_res_y + y) * grid_res_z + z;
          int p2_start = pc2_grid_off[n * G + cell_idx];
          int p2_end;
          if (cell_idx + 1 == grid_total) {
            p2_end = lengths2[n];
          } else {
            p2_end = pc2_grid_off[n * G + cell_idx + 1];
          }

          for (int p2 = p2_start; p2 < p2_end; ++p2) {
            // Compute the squared distance
            float sqdist = 0;
            float diff;
            for (int d = 0; d < D; ++d) {
              diff = points2[n * P2 * D + p2 * D + d] - cur_point[d];
              sqdist += diff * diff;
            }

            if (sqdist <= cur_r2) {
              auto p_idx = sorted_points2_idxs[n * P2 + p2];
              if (p_visible[n * P2 + p_idx] < 1.f)
                p_visible[n * P2 + p_idx] = 1.f;

              // Compute the dot product with the gather point normal
              float dot = 0;
              for (int d = 0; d < D; ++d) {
                dot += cur_normal[d] * p_normal[n * P2 * D + p_idx * D + d];
              }
              if(dot < 0.1) {
                continue;
              }
              // Continue if photon doesn't have the same BSDF as gather point
              if (p_is_glossy[n * P2 + p_idx] != cur_is_glossy)
                continue;
              float photon_rough = p_rough[n * P2 + p_idx];
              if ((photon_rough < (cur_rough - 0.1)) || (photon_rough > (cur_rough + 0.1)))
                continue;
              // Eval BSDF
              float brdf[D];
              if (cur_is_glossy){
                float photon_dir[D];
                for (int d = 0; d < D; ++d) {
                  photon_dir[d] = p_dir[n * P2 * D + p_idx * D + d];
                  brdf[d] = 0.f;
                }
                evalGGX(brdf, cur_dir, photon_dir, cur_albedo, cur_eta, cur_k, cur_rough);
              }
              else{
                evalDiffuse(brdf, cur_albedo);
              }

              for(int d = 0; d < D; ++d) {
                // Compute the flux
                result[d] += p_flux[n * P2 * D + p_idx * D + d] * brdf[d] * M_1_PI * kernel;
              }
            }
          }
        }
      }
    }
    
    // Add the value gathered
    for (int d = 0; d < D; ++d) {
      value[n * P1 * D + old_p1 * D + d] = result[d];
    }
  }
}

std::tuple<at::Tensor, at::Tensor> GatherPPMCuda(
    const at::Tensor points1, const at::Tensor points2,
    const at::Tensor lengths1, const at::Tensor lengths2,
    const at::Tensor pc2_grid_off, const at::Tensor sorted_points1_idxs,
    const at::Tensor sorted_points2_idxs, const at::Tensor params,
    const at::Tensor rs, 
    // Information for the photons and gatherpoint
    const at::Tensor p_normal, const at::Tensor p_dir, const at::Tensor p_rough, 
    const at::Tensor p_is_glossy, const at::Tensor p_flux,
    const at::Tensor g_normal, const at::Tensor g_dir, const at::Tensor g_rough, 
    const at::Tensor g_is_glossy, const at::Tensor g_eta, const at::Tensor g_k,
    const at::Tensor g_albedo
  ) {
  at::TensorArg points1_t{points1, "points1", 1};
  at::TensorArg points2_t{points2, "points2", 2};
  at::TensorArg lengths1_t{lengths1, "lengths1", 3};
  at::TensorArg lengths2_t{lengths2, "lengths2", 4};
  at::TensorArg pc2_grid_off_t{pc2_grid_off, "pc2_grid_off", 5};
  at::TensorArg sorted_points1_idxs_t{sorted_points1_idxs,
                                      "sorted_points1_idxs", 6};
  at::TensorArg sorted_points2_idxs_t{sorted_points2_idxs,
                                      "sorted_points2_idxs", 7};
  at::TensorArg params_t{params, "params", 8};
  at::TensorArg rs_t{rs, "rs", 9};
  at::TensorArg p_normal_t{p_normal, "p_normal", 10};
  at::TensorArg p_dir_t{p_dir, "p_dir", 11};
  at::TensorArg p_rough_t{p_rough, "p_rough", 12};
  at::TensorArg p_is_glossy_t{p_is_glossy, "p_is_glossy", 13};
  at::TensorArg p_flux_t{p_flux, "p_flux", 14};
  at::TensorArg g_normal_t{g_normal, "g_normal", 15};
  at::TensorArg g_dir_t{g_dir, "g_dir", 16};
  at::TensorArg g_rough_t{g_rough, "g_rough", 17};
  at::TensorArg g_is_glossy_t{g_is_glossy, "g_is_glossy", 18};
  at::TensorArg g_eta_t{g_eta, "g_eta", 19};
  at::TensorArg g_k_t{g_k, "g_k", 20};
  at::TensorArg g_albedo_t{g_albedo, "g_albedo", 21};

  at::CheckedFrom c = "GatherPPMCuda";
  at::checkAllSameGPU(
      c, {points1_t, points2_t, lengths1_t, lengths2_t, pc2_grid_off_t,
          sorted_points1_idxs_t, sorted_points2_idxs_t, params_t, rs_t,
          p_normal_t, p_dir_t, p_rough_t, p_is_glossy_t, p_flux_t,
          g_normal_t, g_dir_t, g_rough_t, g_is_glossy_t, g_albedo_t});
  at::checkAllSameType(c, {points1_t, points2_t, params_t, rs_t, p_normal_t, p_dir_t,
                          p_flux_t, g_normal_t, g_dir_t, g_albedo_t});
  at::checkAllSameType(c, {lengths1_t, lengths2_t});
  at::checkAllSameType(c, {pc2_grid_off_t, sorted_points1_idxs_t, sorted_points2_idxs_t});
  at::cuda::CUDAGuard device_guard(points1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int N = points1.size(0);
  int P1 = points1.size(1);
  int D = points1.size(2);
  int P2 = points2.size(1);
  int G = pc2_grid_off.size(1);

  // Return informations: indexes and count
  auto value = at::full({N, P1, 3}, -1, points1.options());

  auto p_visible = at::zeros({N, P2}, points2.options().dtype(at::kFloat));
  
  int threads = 128;
  int blocks = (N * P1 + threads - 1) / threads;

  // Only works in 3D
  GatherKernelV1<<<blocks, threads, 0, stream>>>(
    points1.contiguous().data_ptr<float>(),
    points2.contiguous().data_ptr<float>(),
    lengths1.contiguous().data_ptr<int64_t>(),
    lengths2.contiguous().data_ptr<int64_t>(),
    pc2_grid_off.contiguous().data_ptr<int>(),
    sorted_points1_idxs.contiguous().data_ptr<int>(),
    sorted_points2_idxs.contiguous().data_ptr<int>(),
    params.contiguous().data_ptr<float>(),
    p_normal.contiguous().data_ptr<float>(),
    p_dir.contiguous().data_ptr<float>(),
    p_rough.contiguous().data_ptr<float>(),
    p_is_glossy.contiguous().data_ptr<int>(),
    p_flux.contiguous().data_ptr<float>(),
    p_visible.contiguous().data_ptr<float>(),
    g_normal.contiguous().data_ptr<float>(),
    g_dir.contiguous().data_ptr<float>(),
    g_rough.contiguous().data_ptr<float>(),
    g_is_glossy.contiguous().data_ptr<int>(),
    g_eta.contiguous().data_ptr<float>(),
    g_k.contiguous().data_ptr<float>(),
    g_albedo.contiguous().data_ptr<float>(),
    value.data_ptr<float>(), // Output
    N, P1, P2, G, 
    rs.contiguous().data_ptr<float>());


  return std::make_tuple(value, p_visible);
}
