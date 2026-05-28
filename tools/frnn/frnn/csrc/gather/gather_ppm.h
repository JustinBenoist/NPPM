#include <tuple>

#include "grid/counting_sort.h"
#include "grid/grid.h"
#include "utils/mink.cuh"
// customized dispatch utils for our function type
#include "utils/dispatch.h"
#include "utils/bsdf.h"

// TODO: add docs
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
);
