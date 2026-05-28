#include <tuple>

#include "grid/counting_sort.h"
#include "grid/grid.h"
#include "utils/mink.cuh"
// customized dispatch utils for our function type
#include "utils/dispatch.h"

// TODO: add docs
// Return input encoding and n_matches
std::tuple<at::Tensor, at::Tensor> FindNbrsResamplingTransformCUDA(
    const at::Tensor points1, const at::Tensor points2,
    const at::Tensor lengths1, const at::Tensor lengths2,
    const at::Tensor pc2_grid_off, const at::Tensor sorted_points1_idxs,
    const at::Tensor sorted_points2_idxs, const at::Tensor params, int K,
    const at::Tensor rs, unsigned long long seed,
    const at::Tensor p_normals, const at::Tensor g_normals, const at::Tensor p_flux_avg);
