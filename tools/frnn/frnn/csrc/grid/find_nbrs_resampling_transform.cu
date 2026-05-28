#include "grid/find_nbrs_resampling.h"

#include <curand_kernel.h>

__host__ __device__ inline float Dot(const float* a, const float* b){
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}


__host__ __device__ inline float ToneMapping(float flux) {
  // def tone_mapping_2(flux):
  // a = 0.01
  // log_flux_a = torch.log(flux + a)
  // log_a = math.log(a)
  // num = log_flux_a - log_a
  // denum = log_flux_a - log_a + 1
  // return num / denum
  const float a = 0.01;
  float log_flux_a = logf(flux + a);
  const float log_a = logf(a);
  float num = log_flux_a - log_a;
  float denum = log_flux_a - log_a + 1;
  return num / denum;
}

__global__ void FindNbrsNDKernelV1(
    const float *__restrict__ points1, const float *__restrict__ points2,
    const int64_t *__restrict__ lengths1, const int64_t *__restrict__ lengths2,
    const int *__restrict__ pc2_grid_off,
    const int *__restrict__ sorted_points1_idxs,
    const int *__restrict__ sorted_points2_idxs,
    const float *__restrict__ params,  
    // Output and tmp
    float *__restrict__ input_enc, 
    int32_t *__restrict__ count,
    int32_t *__restrict__ idxs,
    // Other information
    int N, int P1, int P2, int G, int K,
    const float *__restrict__ rs,
    unsigned long long seed,
    const float *__restrict__ p_normal,
    const float *__restrict__ g_normal,
    const float *__restrict__ p_flux_avg) {

  const int D = 3;
  float cur_point[D];
  float cur_normal[D];

  // For the local transform
  float b1[3];
  float b2[3];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, 0, &state);

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
    for (int d = 0; d < D; ++d) {
      // Point get reordered
      cur_point[d] = points1[n * P1 * D + p1 * D + d];
      // Other information are not reordered
      cur_normal[d] = g_normal[n * P1 * D + old_p1 * D + d];
    }

    // Current number of neighbors
    int cur_count = 0;
    
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
    int offset = n * P1 * K + old_p1 * K;
    
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

              float dot = 0;
              for (int d = 0; d < D; ++d) {
                dot += cur_normal[d] * p_normal[n * P2 * D + p_idx * D + d];
              }
              if(dot < 0.1) {
                continue;
              }

              if (cur_count < K) {
                idxs[offset + cur_count] = p2;
                cur_count++;
              } else {
                // Randomly replace
                int r = curand(&state) % (cur_count + 1);
                cur_count++;
                if (r < K) {
                  idxs[offset + r] = p2;
                }
              }
            }
          }
        }
      }
    }

    // Rencode the photons

    // Based on "Building an Orthonormal Basis, Revisited" by
    // Tom Duff, James Burgess, Per Christensen, Christophe Hery, Andrew Kensler,
    // Max Liani, and Ryusuke Villemin
    // https://graphics.pixar.com/library/OrthonormalB/paper.pdf
    float sign = copysignf(1.0f, cur_normal[2]);
    const float a = -1.0f / (sign + cur_normal[2]);
    const float b = cur_normal[0] * cur_normal[1] * a;
    b1[0] = 1.0f + sign * cur_normal[0] * cur_normal[0] * a;
    b1[1] = sign * b;
    b1[2] = -sign * cur_normal[0];
    b2[0] = b;
    b2[1] = sign + cur_normal[1] * cur_normal[1] * a;
    b2[2] = -cur_normal[1];

    int number_photon_selected = min(K, cur_count);
    for(int i = 0; i < number_photon_selected; i++) {
      // Get the photon index
      int id_selected = idxs[offset + i];
      auto p_idx = sorted_points2_idxs[n * P2 + id_selected];
      
      // Get the photon position
      float local_point[3];
      for (int d = 0; d < D; ++d) {
        local_point[d] = points2[n * P2 * D + id_selected * D + d] - cur_point[d];
      }

      // Convert to local coordinates (drop z coordinate)
      input_enc[n * P1 * K * 4 + old_p1 * K * 4 + 4 * i + 0] = Dot(b1, local_point) / cur_r;
      input_enc[n * P1 * K * 4 + old_p1 * K * 4 + 4 * i + 1] = Dot(b2, local_point) / cur_r;
      input_enc[n * P1 * K * 4 + old_p1 * K * 4 + 4 * i + 2] = Dot(cur_normal, local_point) / cur_r;
      input_enc[n * P1 * K * 4 + old_p1 * K * 4 + 4 * i + 3] = ToneMapping(p_flux_avg[n * P2 + p_idx]);
    } 
    // Fill the rest with 0
    // for(int i = cur_count; i < K; i++) {
    //   input_enc[n * P1 * K * 4 + old_p1 * K * 4 + i * K + 0] = 0;
    //   input_enc[n * P1 * K * 4 + old_p1 * K * 4 + i * K + 1] = 0;
    //   input_enc[n * P1 * K * 4 + old_p1 * K * 4 + i * K + 2] = 0;
    //   input_enc[n * P1 * K * 4 + old_p1 * K * 4 + i * K + 3] = 0;
    // }

    // Add the count
    count[n * P1 + old_p1] = cur_count;
  }
}

std::tuple<at::Tensor, at::Tensor> FindNbrsResamplingTransformCUDA(
    const at::Tensor points1, const at::Tensor points2,
    const at::Tensor lengths1, const at::Tensor lengths2,
    const at::Tensor pc2_grid_off, const at::Tensor sorted_points1_idxs,
    const at::Tensor sorted_points2_idxs, const at::Tensor params, int K,
    const at::Tensor rs, unsigned long long seed,
    const at::Tensor p_normals, const at::Tensor g_normals,
    const at::Tensor p_flux_avg) {
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
  at::TensorArg p_normals_t{p_normals, "p_normals", 10};
  at::TensorArg g_normals_t{g_normals, "g_normals", 11};

  at::CheckedFrom c = "FindNbrsResamplingTransformCUDA";
  at::checkAllSameGPU(
      c, {points1_t, points2_t, lengths1_t, lengths2_t, pc2_grid_off_t,
          sorted_points1_idxs_t, sorted_points2_idxs_t, params_t, rs_t,
          p_normals_t, g_normals_t});
  at::checkAllSameType(c, {points1_t, points2_t, params_t, rs_t,
                          p_normals_t, g_normals_t});
  at::checkAllSameType(c, {lengths1_t, lengths2_t});
  at::checkAllSameType(c, {pc2_grid_off_t, sorted_points1_idxs_t, sorted_points2_idxs_t});
  at::cuda::CUDAGuard device_guard(points1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int N = points1.size(0);
  int P1 = points1.size(1);
  int D = points1.size(2);
  int P2 = points2.size(1);
  int G = pc2_grid_off.size(1);

  auto int_dtype = lengths1.options().dtype(at::kInt);

  // Return informations: input_enc and count
  auto input_enc = at::full({N, P1, K, 4}, 0.0, points1.options()).contiguous();
  auto count = at::full({N, P1, 1}, 0, int_dtype);
  // Temp variable
  auto idxs = at::full({N, P1, K}, -1, int_dtype);
  
  int threads = 256;
  int blocks = (N * P1 + threads - 1) / threads;

  // Check if the theta is correct
  // We expect (N, P1, 1)
  if(p_flux_avg.size(0) != N ||
  p_flux_avg.size(1) != P2 ||
  p_flux_avg.size(2) != 1) {
    AT_ERROR("p_flux_avg should be of size (N, P2, 1)");
  }

  
  FindNbrsNDKernelV1<<<blocks, threads, 0, stream>>>(
      points1.contiguous().data_ptr<float>(),
      points2.contiguous().data_ptr<float>(),
      lengths1.contiguous().data_ptr<int64_t>(),
      lengths2.contiguous().data_ptr<int64_t>(),
      pc2_grid_off.contiguous().data_ptr<int>(),
      sorted_points1_idxs.contiguous().data_ptr<int>(),
      sorted_points2_idxs.contiguous().data_ptr<int>(),
      params.contiguous().data_ptr<float>(),
      // Writting
      input_enc.data_ptr<float>(), count.data_ptr<int32_t>(), idxs.data_ptr<int32_t>(), 
      N, P1, P2, G, K, 
      rs.data_ptr<float>(), seed,
      p_normals.contiguous().data_ptr<float>(),
      g_normals.contiguous().data_ptr<float>(),
      p_flux_avg.contiguous().data_ptr<float>());


  return std::make_tuple(input_enc, count);
}
