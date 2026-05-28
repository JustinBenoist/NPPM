#include "gather/gather_ours_gauss.h"

#include <curand_kernel.h>
#include <cuda_fp16.h>
#include "gather/gather_cuda_table.h"

CudaTable GLOBAL_CUDA_TABLE;

__device__ __forceinline__ float get_normalization(
    const cudaTextureObject_t normalization_table,
    const float scale_x,
    const float scale_y) {

      float u = (scale_x - scale_y) * sqrt(2.0) / 2.0;
      float v = (scale_x + scale_y) * sqrt(2.0) / 2.0;
      if (v > 0.0424 * u * u + 0.0868 || sqrtf(scale_x * scale_x + scale_y * scale_y) > 0.09) {
        // bilinear interpolation with x, y in log space
        // where the values from x and y are in the range [TABLE_MIN, TABLE_MAX]
        float x_norm = (logf(scale_x) - TABLE_MIN) / (TABLE_MAX - TABLE_MIN);
        float y_norm = (logf(scale_y) - TABLE_MIN) / (TABLE_MAX - TABLE_MIN);
        return tex2D<float>(normalization_table, x_norm, y_norm);
      } else {
        return 1.f;
      }
}

__device__ inline float Dot(const float* a, const float* b){
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Gathers photons
__global__ void GaussGatherKernel(
    // Gather points
    const float *__restrict__ points1,
    // Photons
    const float *__restrict__ points2,
    const int64_t *__restrict__ lengths1,
    const int64_t *__restrict__ lengths2,
    const int *__restrict__ pc2_grid_off,
    const int *__restrict__ sorted_points1_idxs,
    const int *__restrict__ sorted_points2_idxs,
    const float *__restrict__ params,

    // Photon attributes
    const float *__restrict__ p_normal,
    const float *__restrict__ p_rough,
    const int   *__restrict__ p_is_glossy,

    // Gather attributes
    const float *__restrict__ g_normal,
    const float *__restrict__ g_rough,
    const int   *__restrict__ g_is_glossy,

    // Radius
    const float *__restrict__ rs,

    // Output
    int *__restrict__ nb_matches,
    // photon visibility
    float *__restrict__ p_visible,

    int N, int P1, int P2, int G
) {
    constexpr int D = 3;

    float cur_point[3];
    float cur_normal[3];
    float cur_rough;
    bool  cur_is_glossy;

    int cur_counts = 0;

    int chunks_per_cloud = (P1 + blockDim.x - 1) / blockDim.x;
    int chunks_to_do = N * chunks_per_cloud;

    for (int chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
        int n  = chunk / chunks_per_cloud;
        int p1 = (chunk % chunks_per_cloud) * blockDim.x + threadIdx.x;
        if (p1 >= lengths1[n]) continue;

        int old_p1 = sorted_points1_idxs[n * P1 + p1];

        float cur_r  = rs[n * P1 + old_p1];
        float cur_r2 = cur_r * cur_r;

        cur_rough     = g_rough[n * P1 + old_p1];
        cur_is_glossy = (g_is_glossy[n * P1 + old_p1] == 1);

        for (int d = 0; d < D; ++d) {
            cur_point[d]  = points1[n * P1 * D + p1 * D + d];
            cur_normal[d] = g_normal[n * P1 * D + old_p1 * D + d];
        }

        cur_counts = 0;

        // Grid params
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
            int p2_end   = (cell_idx + 1 == grid_total)
                         ? lengths2[n]
                         : pc2_grid_off[n * G + cell_idx + 1];

            for (int p2 = p2_start; p2 < p2_end; ++p2) {
                int p_idx = sorted_points2_idxs[n * P2 + p2];

                float lx = points2[n * P2 * D + p2 * D + 0] - cur_point[0];
                float ly = points2[n * P2 * D + p2 * D + 1] - cur_point[1];
                float lz = points2[n * P2 * D + p2 * D + 2] - cur_point[2];

                if (lx*lx + ly*ly + lz*lz > cur_r2) continue;

                // mark photon as visible for this batch
                if (p_visible[n * P2 + p_idx] < 1.f) p_visible[n * P2 + p_idx] = 1.f;

                float ndot =
                    cur_normal[0] * p_normal[n * P2 * D + p_idx * D + 0] +
                    cur_normal[1] * p_normal[n * P2 * D + p_idx * D + 1] +
                    cur_normal[2] * p_normal[n * P2 * D + p_idx * D + 2];

                if (ndot < 0.1f) continue;

                // Glossy / diffuse consistency
                if ((p_is_glossy[n * P2 + p_idx] == 1) != cur_is_glossy)
                    continue;

                // Roughness window
                float pr = p_rough[n * P2 + p_idx];
                if (pr < cur_rough - 0.1f || pr > cur_rough + 0.1f)
                    continue;

                cur_counts++;
            }
        }

        nb_matches[n * P1 + old_p1] = cur_counts;
    }
}

// Evaluates the gaussian kernels
__global__ void GaussShadeKernel(
    // Gather points
    const float *__restrict__ points1,
    // Photons
    const float *__restrict__ points2,
    const int64_t *__restrict__ lengths1,
    const int64_t *__restrict__ lengths2,
    const int *__restrict__ pc2_grid_off,
    const int *__restrict__ sorted_points1_idxs,
    const int *__restrict__ sorted_points2_idxs,
    const float *__restrict__ params,

    // Photon attributes
    const float *__restrict__ p_normal,
    const float *__restrict__ p_dir,
    const float *__restrict__ p_rough,
    const int   *__restrict__ p_is_glossy,
    const float *__restrict__ p_flux,

    // Gather attributes
    const float *__restrict__ g_normal,
    const float *__restrict__ g_dir,
    const float *__restrict__ g_rough,
    const int   *__restrict__ g_is_glossy,
    const float *__restrict__ g_eta,
    const float *__restrict__ g_k,
    const float *__restrict__ g_albedo,

    // Radius
    const float *__restrict__ rs,

    // Kernel params
    const half2 *__restrict__ scale,
    const half  *__restrict__ theta,
    const cudaTextureObject_t normalization_table,

    // Output
    float *__restrict__ value,
    float *__restrict__ value_2,

    int N, int P1, int P2, int G
) {
    constexpr int D = 3;

    float cur_point[3];
    float cur_normal[3];
    float cur_dir[3];
    float cur_albedo[3];
    float cur_eta[3];
    float cur_k[3];
    float cur_rough;
    bool  cur_is_glossy;

    float result[3];
    float result_2[3];

    float b1[3], b2[3];

    int chunks_per_cloud = (P1 + blockDim.x - 1) / blockDim.x;
    int chunks_to_do = N * chunks_per_cloud;

    for (int chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
        int n  = chunk / chunks_per_cloud;
        int p1 = (chunk % chunks_per_cloud) * blockDim.x + threadIdx.x;
        if (p1 >= lengths1[n]) continue;

        int old_p1 = sorted_points1_idxs[n * P1 + p1];

        float cur_r  = rs[n * P1 + old_p1];
        float cur_r2 = cur_r * cur_r;

        cur_rough     = g_rough[n * P1 + old_p1];
        cur_is_glossy = (g_is_glossy[n * P1 + old_p1] == 1);

        for (int d = 0; d < D; ++d) {
            cur_point[d]  = points1[n * P1 * D + p1 * D + d];
            cur_normal[d] = g_normal[n * P1 * D + old_p1 * D + d];
            cur_dir[d]    = g_dir[n * P1 * D + old_p1 * D + d];
            cur_albedo[d] = g_albedo[n * P1 * D + old_p1 * D + d];
            cur_eta[d] = g_eta[n * P1 * D + old_p1 * D + d];
            cur_k[d] = g_k[n * P1 * D + old_p1 * D + d];
            result[d]     = 0.0f;
            result_2[d]   = 0.0f;
        }

        // Orthonormal basis
        float sign = copysignf(1.0f, cur_normal[2]);
        float a = -1.0f / (sign + cur_normal[2]);
        float b = cur_normal[0] * cur_normal[1] * a;

        b1[0] = 1.0f + sign * cur_normal[0] * cur_normal[0] * a;
        b1[1] = sign * b;
        b1[2] = -sign * cur_normal[0];

        b2[0] = b;
        b2[1] = sign + cur_normal[1] * cur_normal[1] * a;
        b2[2] = -cur_normal[1];

        // Gaussian kernel params
        float2 sc = __half22float2(scale[n * P1 + old_p1]);
        sc.x = fminf(fmaxf(sc.x, 1e-5f), sqrtf(3.f));
        sc.y = fminf(fmaxf(sc.y, 1e-5f), sqrtf(3.f));
        sc.x *= sc.x;
        sc.y *= sc.y;

        float theta_v = __half2float(theta[n * P1 + old_p1]);
        float s, c;
        __sincosf(theta_v, &s, &c);

        float cov00 = c*c*sc.x + s*s*sc.y + 1e-5f;
        float cov01 = c*s*(sc.x - sc.y);
        float cov11 = s*s*sc.x + c*c*sc.y + 1e-5f;

        float det = cov00 * cov11 - cov01 * cov01;
        float norm = 1.f / (2.f * M_PI * sqrtf(det));
        // Get normalization factor from the lookup table
        float norm_factor =
            1.f / (get_normalization(normalization_table, sc.x, sc.y) * cur_r2 + 1e-6f);

        // Grid
        float gx = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_X];
        float gy = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_Y];
        float gz = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_MIN_Z];
        float gd = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_DELTA];
        int rx   = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_X];
        int ry   = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_Y];
        int rz   = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_RES_Z];
        int gt   = params[n * GRID_3D_PARAMS_SIZE + GRID_3D_TOTAL];

        int minx = max(0, int((cur_point[0] - gx - cur_r) * gd));
        int maxx = min(rx - 1, int((cur_point[0] - gx + cur_r) * gd));
        int miny = max(0, int((cur_point[1] - gy - cur_r) * gd));
        int maxy = min(ry - 1, int((cur_point[1] - gy + cur_r) * gd));
        int minz = max(0, int((cur_point[2] - gz - cur_r) * gd));
        int maxz = min(rz - 1, int((cur_point[2] - gz + cur_r) * gd));

        for (int x = minx; x <= maxx; ++x)
        for (int y = miny; y <= maxy; ++y)
        for (int z = minz; z <= maxz; ++z) {

            int cell = (x * ry + y) * rz + z;
            int s2 = pc2_grid_off[n * G + cell];
            int e2 = (cell + 1 == gt) ? lengths2[n]
                                      : pc2_grid_off[n * G + cell + 1];

            for (int p2 = s2; p2 < e2; ++p2) {
                int p_idx = sorted_points2_idxs[n * P2 + p2];

                float lp[3] = {
                    points2[n * P2 * D + p2 * D + 0] - cur_point[0],
                    points2[n * P2 * D + p2 * D + 1] - cur_point[1],
                    points2[n * P2 * D + p2 * D + 2] - cur_point[2]
                };

                if (lp[0]*lp[0] + lp[1]*lp[1] + lp[2]*lp[2] > cur_r2)
                    continue;

                float nd =
                    cur_normal[0] * p_normal[n * P2 * D + p_idx * D + 0] +
                    cur_normal[1] * p_normal[n * P2 * D + p_idx * D + 1] +
                    cur_normal[2] * p_normal[n * P2 * D + p_idx * D + 2];

                if (nd < 0.1f) continue;

                // Continue if photon doesn't have the same BSDF as gather point
                if ((p_is_glossy[n * P2 + p_idx] == 1) != cur_is_glossy) continue;
                float pr = p_rough[n * P2 + p_idx];
                if (pr < cur_rough - 0.1f || pr > cur_rough + 0.1f) continue;
                
                // Eval BSDF
                float brdf[3];
                if (cur_is_glossy) {
                    float pd[3];
                    for (int d = 0; d < 3; ++d)
                        pd[d] = p_dir[n * P2 * D + p_idx * D + d];
                    evalGGX(brdf, cur_dir, pd, cur_albedo, cur_eta, cur_k, cur_rough);
                } else {
                    evalDiffuse(brdf, cur_albedo);
                }

                float u = (b1[0]*lp[0] + b1[1]*lp[1] + b1[2]*lp[2]) / cur_r;
                float v = (b2[0]*lp[0] + b2[1]*lp[1] + b2[2]*lp[2]) / cur_r;

                float w = norm * expf(
                    -(0.5f / det) * (cov11*u*u + cov00*v*v - 2.f*cov01*u*v));
                
                // Compute flux
                for (int d = 0; d < 3; ++d) {
                    float f = p_flux[n * P2 * D + p_idx * D + d] * brdf[d] * M_1_PI;
                    result[d]   += f * w * norm_factor;
                    result_2[d] += f;
                }
            }
        }

        for (int d = 0; d < 3; ++d) {
            value[n * P1 * D + old_p1 * D + d]   = result[d];
            value_2[n * P1 * D + old_p1 * D + d] = result_2[d];
        }
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> GatherOursGaussCuda(
  const at::Tensor points1, const at::Tensor points2,
  const at::Tensor lengths1, const at::Tensor lengths2,
  const at::Tensor pc2_grid_off, const at::Tensor sorted_points1_idxs,
  const at::Tensor sorted_points2_idxs, const at::Tensor params,
  const at::Tensor rs,
  const at::Tensor nb_matches,
  // Information for the photons and gatherpoint
  const at::Tensor p_normal, const at::Tensor p_dir, const at::Tensor p_rough, 
  const at::Tensor p_is_glossy, const at::Tensor p_flux,
  const at::Tensor g_normal, const at::Tensor g_dir, const at::Tensor g_rough, 
  const at::Tensor g_is_glossy, const at::Tensor g_eta, const at::Tensor g_k,
  const at::Tensor g_albedo,
  // Information specific to the Gaussian kernel and normalization table
  const at::Tensor scale, const at::Tensor theta
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
  at::TensorArg nb_matches_t{rs, "nb_matches", 10};
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
  at::TensorArg p_normal_t{p_normal, "p_normal", 22};
  at::TensorArg scale_t{scale, "scale", 23};
  at::TensorArg theta_t{theta, "theta", 24};

  at::CheckedFrom c = "GatherOursGaussCuda";
  at::checkAllSameGPU(
      c, {points1_t, points2_t, lengths1_t, lengths2_t, pc2_grid_off_t,
          sorted_points1_idxs_t, sorted_points2_idxs_t, params_t, rs_t,
          p_normal_t, p_dir_t, p_rough_t, p_is_glossy_t, p_flux_t,
          g_normal_t, g_dir_t, g_rough_t, g_is_glossy_t, g_eta_t, g_k_t, g_albedo_t,
          scale_t, theta_t});
  at::checkAllSameType(c, {points1_t, points2_t, params_t, rs_t, p_normal_t,
                          p_flux_t, g_normal_t, g_albedo_t});
  at::checkAllSameType(c, {lengths1_t, lengths2_t});
  at::checkAllSameType(c, {pc2_grid_off_t, sorted_points1_idxs_t, sorted_points2_idxs_t});
  at::cuda::CUDAGuard device_guard(points1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // TODO: Check the half precision
  // scale_t, theta_t,

  int N = points1.size(0);
  int P1 = points1.size(1);
  int D = points1.size(2);
  int P2 = points2.size(1);
  int G = pc2_grid_off.size(1);

  // Return informations: indexes and count
  auto value = at::full({N, P1, 3}, -1, points1.options());
  auto value_2 = at::full({N, P1, 3}, -1, points1.options());
  
  int threads = 64;
  int blocks = (N * P1 + threads - 1) / threads;

  // Check if the scale is correct
  // We expect (N, P1, 2)
  if(scale.size(0) != N ||
    scale.size(1) != P1 ||
    scale.size(2) != 2) {
    AT_ERROR("scale should be of size (N, P1, 2)");
  }

  // Check if the theta is correct
  // We expect (N, P1, 1)
  if(theta.size(0) != N ||
    theta.size(1) != P1 ||
    theta.size(2) != 1) {
    AT_ERROR("theta should be of size (N, P1, 1)");
  }

  // Only works in 3D
    auto p_visible = at::zeros({N, P2}, points2.options().dtype(at::kFloat));

    GaussGatherKernel<<<blocks, threads, 0, stream>>>(
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
        nb_matches.contiguous().data_ptr<int>(),
        p_visible.contiguous().data_ptr<float>(),
        N, P1, P2, G
  );
  
  GaussShadeKernel<<<blocks, threads, 0, stream>>>(
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
    g_normal.contiguous().data_ptr<float>(),
    g_dir.contiguous().data_ptr<float>(),
    g_rough.contiguous().data_ptr<float>(),
    g_is_glossy.contiguous().data_ptr<int>(),
    g_eta.contiguous().data_ptr<float>(),
    g_k.contiguous().data_ptr<float>(),
    g_albedo.contiguous().data_ptr<float>(),
    rs.contiguous().data_ptr<float>(),
    (half2*)scale.contiguous().data_ptr<at::Half>(),
    (half*)theta.contiguous().data_ptr<at::Half>(),
    GLOBAL_CUDA_TABLE.tex,
    value.data_ptr<float>(),
    value_2.data_ptr<float>(), // Output
    N, P1, P2, G
  );


    return std::make_tuple(value, value_2, p_visible);
}
