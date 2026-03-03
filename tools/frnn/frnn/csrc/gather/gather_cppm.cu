#include "gather/gather_cppm.h"

#include <curand_kernel.h>

// Gathers photons
__global__ void CPPMGatherKernel(
    const float* __restrict__ points1,
    const float* __restrict__ points2,
    const int64_t* __restrict__ lengths1,
    const int64_t* __restrict__ lengths2,
    const int* __restrict__ pc2_grid_off,
    const int* __restrict__ sorted_points1_idxs,
    const int* __restrict__ sorted_points2_idxs,
    const float* __restrict__ params,
    const float* __restrict__ p_normal,
    const float* __restrict__ p_rough,
    const int*   __restrict__ p_is_glossy,
    const float* __restrict__ g_normal,
    const float* __restrict__ g_rough,
    const int*   __restrict__ g_is_glossy,

    float* __restrict__ rs,
    int*   __restrict__ total_counts,
    int*   __restrict__ min_counts,
    int*   __restrict__ photon_counts,

    int N, int P1, int P2, int G
) {
    constexpr int D = 3;

    float cur_point[3];
    float cur_normal[3];
    float cur_rough;
    bool  cur_is_glossy;

    int   cur_total;
    int   cur_sector[SEC_U * SEC_V];
    bool  passed[SEC_U];

    int chunks_per_cloud = (P1 + blockDim.x - 1) / blockDim.x;
    int chunks_to_do = N * chunks_per_cloud;

    for (int chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
        int n = chunk / chunks_per_cloud;
        int p1 = (chunk % chunks_per_cloud) * blockDim.x + threadIdx.x;
        if (p1 >= lengths1[n]) continue;

        int old_p1 = sorted_points1_idxs[n * P1 + p1];

        float cur_r = rs[n * P1 + old_p1];
        float cur_r2 = cur_r * cur_r;

        for (int d = 0; d < D; ++d) {
            cur_point[d]  = points1[n * P1 * D + p1 * D + d];
            cur_normal[d] = g_normal[n * P1 * D + old_p1 * D + d];
        }

        cur_rough = g_rough[n * P1 + old_p1];
        cur_is_glossy = g_is_glossy[n * P1 + old_p1];

        cur_total = total_counts[n * P1 + old_p1];
        for (int i = 0; i < SEC_U * SEC_V; ++i)
            cur_sector[i] = photon_counts[n * P1 * SEC_U * SEC_V + old_p1 * SEC_U * SEC_V + i];

        // Grid bounds
        float grid_min_x = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_X];
        float grid_min_y = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_Y];
        float grid_min_z = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_Z];
        float grid_delta = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_DELTA];
        int grid_res_x   = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_X];
        int grid_res_y   = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_Y];
        int grid_res_z   = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_Z];
        int grid_total   = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_TOTAL];

        int min_x = max(0, int((cur_point[0] - grid_min_x - cur_r) * grid_delta));
        int max_x = min(grid_res_x - 1, int((cur_point[0] - grid_min_x + cur_r) * grid_delta));
        int min_y = max(0, int((cur_point[1] - grid_min_y - cur_r) * grid_delta));
        int max_y = min(grid_res_y - 1, int((cur_point[1] - grid_min_y + cur_r) * grid_delta));
        int min_z = max(0, int((cur_point[2] - grid_min_z - cur_r) * grid_delta));
        int max_z = min(grid_res_z - 1, int((cur_point[2] - grid_min_z + cur_r) * grid_delta));

        for (int x = min_x; x <= max_x; ++x)
        for (int y = min_y; y <= max_y; ++y)
        for (int z = min_z; z <= max_z; ++z) {
            int cell = (x * grid_res_y + y) * grid_res_z + z;
            int p2_start = pc2_grid_off[n * G + cell];
            int p2_end   = (cell + 1 == grid_total)
                           ? lengths2[n]
                           : pc2_grid_off[n * G + cell + 1];
            
            // Count photons in sectors
            for (int p2 = p2_start; p2 < p2_end; ++p2) {
                int idx = sorted_points2_idxs[n * P2 + p2];

                float dx = points2[n * P2 * 3 + p2 * 3 + 0] - cur_point[0];
                float dy = points2[n * P2 * 3 + p2 * 3 + 1] - cur_point[1];
                float dz = points2[n * P2 * 3 + p2 * 3 + 2] - cur_point[2];
                if (dx*dx + dy*dy + dz*dz > cur_r2) continue;

                if (p_is_glossy[n * P2 + idx] != cur_is_glossy) continue;

                float pr = p_rough[n * P2 + idx];
                if (fabsf(pr - cur_rough) > 0.1f) continue;

                cur_total++;
                int sec = getSection(dx, dy, dz, cur_r);
                cur_sector[sec]++;
            }
        }
        // Perform chi² test and update radius
        if (cur_total > min_counts[n * P1 + old_p1]) {
            chiSquaredTest(cur_sector, passed);
            if (!passed[SEC_U - 1]) {
                min_counts[n * P1 + old_p1] *= BETA;
                rs[n * P1 + old_p1] *= 0.9f;
                cur_total = 0;
                for (int i = 0; i < SEC_U * SEC_V; ++i)
                    cur_sector[i] = 0;
            }
        }

        total_counts[n * P1 + old_p1] = cur_total;
        for (int i = 0; i < SEC_U * SEC_V; ++i)
            photon_counts[n * P1 * SEC_U * SEC_V + old_p1 * SEC_U * SEC_V + i] = cur_sector[i];
    }
}

// Evaluates flux
__global__ void CPPMShadeKernel(
    // Gather points
    const float *__restrict__ points1,
    // Photons
    const float *__restrict__ points2,
    const int64_t *__restrict__ lengths1,
    const int *__restrict__ pc2_grid_off,
    const int *__restrict__ sorted_points1_idxs,
    const int *__restrict__ sorted_points2_idxs,
    const float *__restrict__ params,

    // Photon data
    const float *__restrict__ p_normal,
    const float *__restrict__ p_dir,
    const float *__restrict__ p_rough,
    const int   *__restrict__ p_is_glossy,
    const float *__restrict__ p_flux,

    // Gather point data
    const float *__restrict__ g_normal,
    const float *__restrict__ g_dir,
    const float *__restrict__ g_rough,
    const int   *__restrict__ g_is_glossy,
    const float *__restrict__ g_eta,
    const float *__restrict__ g_k,
    const float *__restrict__ g_albedo,

    // Radius (already updated by gather kernel)
    const float *__restrict__ rs,

    // Output
    float *__restrict__ value,

    int N, int P1, int P2, int G
) {
    constexpr int D = 3;

    float cur_point[3];
    float cur_normal[3];
    float cur_dir[3];
    float cur_eta[3];
    float cur_k[3];
    float cur_albedo[3];
    float result[3];

    int chunks_per_cloud = (P1 + blockDim.x - 1) / blockDim.x;
    int chunks_to_do = N * chunks_per_cloud;

    for (int chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
        int n = chunk / chunks_per_cloud;
        int p1 = (chunk % chunks_per_cloud) * blockDim.x + threadIdx.x;
        if (p1 >= lengths1[n]) continue;

        int old_p1 = sorted_points1_idxs[n * P1 + p1];

        float cur_r = rs[n * P1 + old_p1];
        float cur_r2 = cur_r * cur_r;
        float kernel = 1.0f / (M_PI * cur_r2);

        for (int d = 0; d < D; ++d) {
            cur_point[d]  = points1[n * P1 * D + p1 * D + d];
            cur_normal[d] = g_normal[n * P1 * D + old_p1 * D + d];
            cur_dir[d]    = g_dir[n * P1 * D + old_p1 * D + d];
            cur_albedo[d] = g_albedo[n * P1 * D + old_p1 * D + d];
            cur_eta[d]    = g_eta[n * P1 * D + old_p1 * D + d];
            cur_k[d]    = g_k[n * P1 * D + old_p1 * D + d];
            result[d]     = 0.0f;
        }

        float cur_rough = g_rough[n * P1 + old_p1];
        bool  cur_is_glossy = g_is_glossy[n * P1 + old_p1];

        // Grid parameters
        float grid_min_x = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_X];
        float grid_min_y = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_Y];
        float grid_min_z = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_Z];
        float grid_delta = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_DELTA];
        int grid_res_x   = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_X];
        int grid_res_y   = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_Y];
        int grid_res_z   = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_Z];
        int grid_total   = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_TOTAL];

        int min_x = max(0, int((cur_point[0] - grid_min_x - cur_r) * grid_delta));
        int max_x = min(grid_res_x - 1, int((cur_point[0] - grid_min_x + cur_r) * grid_delta));
        int min_y = max(0, int((cur_point[1] - grid_min_y - cur_r) * grid_delta));
        int max_y = min(grid_res_y - 1, int((cur_point[1] - grid_min_y + cur_r) * grid_delta));
        int min_z = max(0, int((cur_point[2] - grid_min_z - cur_r) * grid_delta));
        int max_z = min(grid_res_z - 1, int((cur_point[2] - grid_min_z + cur_r) * grid_delta));

        for (int x = min_x; x <= max_x; ++x)
        for (int y = min_y; y <= max_y; ++y)
        for (int z = min_z; z <= max_z; ++z) {

            int cell_idx = (x * grid_res_y + y) * grid_res_z + z;
            int p2_start = pc2_grid_off[n * G + cell_idx];
            int p2_end = (cell_idx + 1 == grid_total)
                       ? lengths1[n]
                       : pc2_grid_off[n * G + cell_idx + 1];

            for (int p2 = p2_start; p2 < p2_end; ++p2) {
                int p_idx = sorted_points2_idxs[n * P2 + p2];

                float dx = points2[n * P2 * D + p2 * D + 0] - cur_point[0];
                float dy = points2[n * P2 * D + p2 * D + 1] - cur_point[1];
                float dz = points2[n * P2 * D + p2 * D + 2] - cur_point[2];
                if (dx*dx + dy*dy + dz*dz > cur_r2) continue;

                float ndot = cur_normal[0]*p_normal[n * P2 * D + p_idx * D + 0] +
                             cur_normal[1]*p_normal[n * P2 * D + p_idx * D + 1] +
                             cur_normal[2]*p_normal[n * P2 * D + p_idx * D + 2];
                if (ndot < 0.1f) continue;

                // Continue if photon doesn't have the same BSDF as gather point
                if (p_is_glossy[n * P2 + p_idx] != cur_is_glossy) continue;
                float pr = p_rough[n * P2 + p_idx];
                if (fabsf(pr - cur_rough) > 0.1f) continue;
                
                // Eval BSDF
                float brdf[3];
                if (cur_is_glossy) {
                    float photon_dir[3];
                    photon_dir[0] = p_dir[n * P2 * D + p_idx * D + 0];
                    photon_dir[1] = p_dir[n * P2 * D + p_idx * D + 1];
                    photon_dir[2] = p_dir[n * P2 * D + p_idx * D + 2];
                    evalGGX(brdf, cur_dir, photon_dir, cur_albedo, cur_eta, cur_k, cur_rough);
                } else {
                    evalDiffuse(brdf, cur_albedo);
                }
                
                // Add flux
                for (int d = 0; d < D; ++d) {
                    result[d] += p_flux[n * P2 * D + p_idx * D + d]
                               * brdf[d]
                               * kernel * M_1_PI;
                }
            }
        }

        for (int d = 0; d < D; ++d)
            value[n * P1 * D + old_p1 * D + d] = result[d];
    }
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> GatherCPPMCuda(
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
    const at::Tensor g_albedo,
    // Information specific to CPPM
    const at::Tensor total_counts,
    const at::Tensor min_counts,
    const at::Tensor photon_counts
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
  // CPPM specific
  at::TensorArg total_counts_t{total_counts, "total_counts", 22};
  at::TensorArg min_counts_t{min_counts, "min_counts", 23};
  at::TensorArg photon_counts_t{photon_counts, "photon_counts", 24};

  at::CheckedFrom c = "GatherCPPMCuda";
  at::checkAllSameGPU(
      c, {points1_t, points2_t, lengths1_t, lengths2_t, pc2_grid_off_t,
          sorted_points1_idxs_t, sorted_points2_idxs_t, params_t, rs_t,
          p_normal_t, p_dir_t, p_rough_t, p_is_glossy_t, p_flux_t,
          g_normal_t, g_dir_t, g_rough_t, g_is_glossy_t, g_eta_t, g_k_t, g_albedo_t, 
          total_counts_t, min_counts_t, photon_counts_t});
  // float
  at::checkAllSameType(c, {points1_t, points2_t, params_t, rs_t, p_normal_t,
                          p_flux_t, g_normal_t, g_albedo_t});
  // int64                                                  
  at::checkAllSameType(c, {lengths1_t, lengths2_t});
  // int32
  at::checkAllSameType(c, {pc2_grid_off_t, sorted_points1_idxs_t, sorted_points2_idxs_t, total_counts_t, min_counts_t, photon_counts_t});
  at::cuda::CUDAGuard device_guard(points1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int N = points1.size(0);
  int P1 = points1.size(1);
  int D = points1.size(2);
  int P2 = points2.size(1);
  int G = pc2_grid_off.size(1);

  // Check if the photon count have correct size
  // We expect (N, P1, SEC_U * SEC_V)
  if(photon_counts.size(0) != N ||
     photon_counts.size(1) != P1 ||
     photon_counts.size(2) != SEC_U * SEC_V) {
    AT_ERROR("photon_counts should be of size (N, P1, SEC_U * SEC_V)");
  }

  // Return informations: indexes and count
  auto value = at::full({N, P1, 3}, -1, points1.options());
  
  int threads = 128;
  int blocks = (N * P1 + threads - 1) / threads;

  // Only works in 3D
  CPPMGatherKernel<<<blocks, threads, 0, stream>>>(
    points1.contiguous().data_ptr<float>(),
    points2.contiguous().data_ptr<float>(),
    lengths1.contiguous().data_ptr<int64_t>(),
    lengths2.contiguous().data_ptr<int64_t>(),
    pc2_grid_off.contiguous().data_ptr<int>(),
    sorted_points1_idxs.contiguous().data_ptr<int>(),
    sorted_points2_idxs.contiguous().data_ptr<int>(),
    params.contiguous().data_ptr<float>(),
    p_normal.contiguous().data_ptr<float>(),
    p_rough.contiguous().data_ptr<float>(),
    p_is_glossy.contiguous().data_ptr<int>(),
    g_normal.contiguous().data_ptr<float>(),
    g_rough.contiguous().data_ptr<float>(),
    g_is_glossy.contiguous().data_ptr<int>(),
    rs.data_ptr<float>(), // Output
    // CPPM specific
    total_counts.contiguous().data_ptr<int>(),
    min_counts.contiguous().data_ptr<int>(),
    photon_counts.contiguous().data_ptr<int>(),
    N, P1, P2, G);

  CPPMShadeKernel<<<blocks, threads, 0, stream>>>(
    points1.contiguous().data_ptr<float>(),
    points2.contiguous().data_ptr<float>(),
    lengths1.contiguous().data_ptr<int64_t>(),
    pc2_grid_off.contiguous().data_ptr<int>(),
    sorted_points1_idxs.contiguous().data_ptr<int>(),
    sorted_points2_idxs.contiguous().data_ptr<int>(),
    params.contiguous().data_ptr<float>(),
    p_normal.contiguous().data_ptr<float>(),
    p_dir.contiguous().data_ptr<float>(),
    p_rough.contiguous().data_ptr<float>(),
    p_is_glossy.contiguous().data_ptr<int>(),
    p_flux.contiguous().data_ptr<float>(),
    g_normal.contiguous().data_ptr<float>(),
    g_dir.contiguous().data_ptr<float>(),
    g_rough.contiguous().data_ptr<float>(),
    g_is_glossy.contiguous().data_ptr<int>(),
    g_eta.contiguous().data_ptr<float>(),
    g_k.contiguous().data_ptr<float>(),
    g_albedo.contiguous().data_ptr<float>(),
    rs.data_ptr<float>(),
    value.data_ptr<float>(), // Output
    N, P1, P2, G);


  return std::make_tuple(value, total_counts, min_counts, photon_counts);
}
