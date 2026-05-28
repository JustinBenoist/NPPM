#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <float.h>
#include <iostream>
#include <tuple>

#include "utils/dispatch.cuh"

// Up to 8 dim
constexpr int V2_MIN_D = 1;
constexpr int V2_MAX_D = 8;
// Up to 64 neighbors
constexpr int V2_MIN_K = 1;
constexpr int V2_MAX_K = 64;

////////////////// Resampling
#include <curand_kernel.h>
// D: dimension
// K: number of neighbors
template <typename scalar_t, int64_t D, int64_t K>
__global__ void FRNNBruteForceResamplingKernel(const scalar_t *__restrict__ points1,
                                     const scalar_t *__restrict__ points2,
                                     const int64_t *__restrict__ lengths1,
                                     const int64_t *__restrict__ lengths2,
                                     int64_t *__restrict__ idxs, 
                                     int64_t *__restrict__ count,
                                     int N, 
                                     int P1,
                                     int P2, 
                                     float r2, // Radius squared
                                     unsigned long long seed
                                    ) {
  // Seed with idx from a 2D kernel
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, 0, &state);

  // The points founds (K) -- per point
  int min_idxs[K];
  scalar_t cur_point[D];  
  int cur_count = 0;

  // Check for different chuncking
  int chunks_per_cloud = (1 + (P1 - 1) / blockDim.x);
  int chunks_to_do = N * chunks_per_cloud;
  for (int chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    // Index of the point cloud
    int n = chunk / chunks_per_cloud;
    // Get where to start the point cloud
    int start_point = blockDim.x * (chunk % chunks_per_cloud);

    // The current points we consider
    int p1 = start_point + threadIdx.x;
    if (p1 >= lengths1[n]) continue;

    // The point we consider
    for (int d = 0; d < D; ++d) {
      cur_point[d] = points1[n * P1 * D + p1 * D + d];
    }
    
    // Reservoir sampling
    int cur_count = 0;

    // Go through all the second array
    int length2 = lengths2[n];
    for (int p2 = 0; p2 < length2; ++p2) {
      
      // Distance squared
      scalar_t dist = 0;
      for (int d = 0; d < D; ++d) {
        int offset = n * P2 * D + p2 * D + d;
        scalar_t diff = cur_point[d] - points2[offset];
        dist += diff * diff;
      }
      
      // If outside, quit
      if (dist >= r2) continue;
      
      // Simple reservoir sampling
      if(cur_count < K) {
        // Add to the list
        min_idxs[cur_count] = p2;
        cur_count++;
      } else {
        // Reservoir sampling
        cur_count++; // Increment the number of pushed points
        int r = curand(&state) % cur_count;
        if (r < K) {
          min_idxs[r] = p2;
        }
      }
    }

    count[n * P1 + p1] = cur_count;
    if(cur_count > K) {
      cur_count = K;
    }

    for (int k = 0; k < cur_count; ++k) {
      // Copy all values
      idxs[n * P1 * K + p1 * K + k] = min_idxs[k];
    }
  }
}

// This is a shim so we can dispatch using DispatchKernel2D
template <typename scalar_t, int64_t D, int64_t K>
struct FRNNBruteForceResamplingFunctor {
  static void run(int blocks, int threads, const scalar_t *__restrict__ points1,
                  const scalar_t *__restrict__ points2,
                  const int64_t *__restrict__ lengths1,
                  const int64_t *__restrict__ lengths2,
                  int64_t *__restrict__ idxs, int64_t *__restrict__ count,
                  int N, int P1, int P2, float r2, unsigned long long seed) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    FRNNBruteForceResamplingKernel<scalar_t, D, K><<<blocks, threads, 0, stream>>>(
        points1, points2, lengths1, lengths2, idxs, count, N, P1, P2, r2, seed);
  }
};

std::tuple<at::Tensor, at::Tensor> FRNNBruteForceResamplingCUDA(
    const at::Tensor &p1, const at::Tensor &p2, 
    const at::Tensor &lengths1,
    const at::Tensor &lengths2, 
    int K, 
    float r,
    unsigned long long seed) {
  // Check inputs are on the same device
  at::TensorArg p1_t{p1, "p1", 1}, p2_t{p2, "p2", 2},
      lengths1_t{lengths1, "lengths1", 3}, lengths2_t{lengths2, "lengths2", 4};
  at::CheckedFrom c = "FRNNBruteForceResamplingCUDA";
  at::checkAllSameGPU(c, {p1_t, p2_t, lengths1_t, lengths2_t});
  at::checkAllSameType(c, {p1_t, p2_t});
  at::checkAllSameType(c, {lengths1_t, lengths2_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(p1.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto N = p1.size(0);
  auto P1 = p1.size(1);
  auto P2 = p2.size(1);
  auto D = p2.size(2);
  int64_t K_64 = K;
  float r2 = r * r;

  TORCH_CHECK(p2.size(2) == D, "Point sets must have the same last dimension");
  auto long_dtype = lengths1.options().dtype(at::kLong);
  auto idxs = at::full({N, P1, K}, -1, long_dtype);
  auto count = at::full({N, P1, 1}, -1, long_dtype); // Init to -1
  

  if (idxs.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(idxs, count);
  }

  AT_ASSERTM(D >= V2_MIN_D && D <= V2_MAX_D && K >= V2_MIN_K && D <= V2_MAX_K,
             "Invalid range for K or D");

  int threads = 256;
  int blocks = 256;
  AT_DISPATCH_FLOATING_TYPES(
      p1.scalar_type(), "frnn_kernel_cuda", ([&] {
        DispatchKernel2D<FRNNBruteForceResamplingFunctor, scalar_t, V2_MIN_D, V2_MAX_D,
                         V2_MIN_K, V2_MAX_K>(
            D, K_64, blocks, threads, p1.contiguous().data_ptr<scalar_t>(),
            p2.contiguous().data_ptr<scalar_t>(),
            lengths1.contiguous().data_ptr<int64_t>(),
            lengths2.contiguous().data_ptr<int64_t>(),
            idxs.data_ptr<int64_t>(), count.data_ptr<int64_t>(), N, P1, P2,
            r2, seed);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(idxs, count);
}