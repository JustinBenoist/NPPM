#include "grid/find_nbrs_resampling.h"

#include <curand_kernel.h>

template <int D>
__global__ void FindNbrsNDKernelV1(
    const float *__restrict__ points1, const float *__restrict__ points2,
    const int64_t *__restrict__ lengths1, const int64_t *__restrict__ lengths2,
    const int *__restrict__ pc2_grid_off,
    const int *__restrict__ sorted_points1_idxs,
    const int *__restrict__ sorted_points2_idxs,
    const float *__restrict__ params,  
    //float *__restrict__ dists,
    int32_t *__restrict__ idxs, 
    int32_t *__restrict__ count,
    int N, int P1, int P2, int G, int K,
    const float *__restrict__ rs,
    unsigned long long seed) {
  float cur_point[D];

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
      cur_point[d] = points1[n * P1 * D + p1 * D + d];
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
              if (cur_count < K) {
                idxs[offset + cur_count] = sorted_points2_idxs[n * P2 + p2];
                cur_count++;
              } else {
                // Randomly replace
                int r = curand(&state) % (cur_count + 1);
                cur_count++;
                if (r < K) {
                  idxs[offset + r] = sorted_points2_idxs[n * P2 + p2];
                }
              }
            }
          }
        }
      }
    }
    
    // Add the count
    count[n * P1 + old_p1] = cur_count;
  }
}

template <int D>
struct FindNbrsKernelV1Functor {
  static void run(int blocks, int threads,
                  const float *__restrict__ points1,            // (N, P1, D)
                  const float *__restrict__ points2,            // (N, P2, D)
                  const int64_t *__restrict__ lengths1,            // (N,)
                  const int64_t *__restrict__ lengths2,            // (N,)
                  const int *__restrict__ pc2_grid_off,         // (N, G)
                  const int *__restrict__ sorted_points1_idxs,  // (N, P)
                  const int *__restrict__ sorted_points2_idxs,  // (N, P)
                  const float *__restrict__ params,             // (N,)
                  // float *__restrict__ dists,                    // (N, P1, K)
                  int32_t *__restrict__ idxs,                      // (N, P1, K)
                  int32_t *__restrict__ count,                    // (N, P1, 1)
                  int N, int P1, int P2, int G, int K, const float *rs,
                  unsigned long long seed) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    FindNbrsNDKernelV1<D><<<blocks, threads, 0, stream>>>(
        points1, points2, lengths1, lengths2, pc2_grid_off,
        sorted_points1_idxs, sorted_points2_idxs, params, idxs, count, N, P1,
        P2, G, K, rs, seed);
  }
};

std::tuple<at::Tensor, at::Tensor> FindNbrsResamplingCUDA(
    const at::Tensor points1, const at::Tensor points2,
    const at::Tensor lengths1, const at::Tensor lengths2,
    const at::Tensor pc2_grid_off, const at::Tensor sorted_points1_idxs,
    const at::Tensor sorted_points2_idxs, const at::Tensor params, int K,
    const at::Tensor rs, unsigned long long seed) {
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

  at::CheckedFrom c = "FindNbrsResamplingCUDA";
  at::checkAllSameGPU(
      c, {points1_t, points2_t, lengths1_t, lengths2_t, pc2_grid_off_t,
          sorted_points1_idxs_t, sorted_points2_idxs_t, params_t, rs_t});
  at::checkAllSameType(c, {points1_t, points2_t, params_t, rs_t});
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

  // Return informations: indexes and count
  auto idxs = at::full({N, P1, K}, -1, int_dtype);
  auto count = at::full({N, P1, 1}, -1, int_dtype);
  
  int threads = 256;
  int blocks = (N * P1 + threads - 1) / threads;

  
  DispatchKernel1D<FindNbrsKernelV1Functor, V1_MIN_D, V1_MAX_D>(
      D, blocks, threads, points1.contiguous().data_ptr<float>(),
      points2.contiguous().data_ptr<float>(),
      lengths1.contiguous().data_ptr<int64_t>(),
      lengths2.contiguous().data_ptr<int64_t>(),
      pc2_grid_off.contiguous().data_ptr<int>(),
      sorted_points1_idxs.contiguous().data_ptr<int>(),
      sorted_points2_idxs.contiguous().data_ptr<int>(),
      params.contiguous().data_ptr<float>(),
      idxs.data_ptr<int32_t>(), count.data_ptr<int32_t>(), N, P1, P2, G, K, rs.data_ptr<float>(), seed);


  return std::make_tuple(idxs, count);
}
