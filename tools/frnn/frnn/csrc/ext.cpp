#include <torch/extension.h>

#include "backward/backward.h"
#include "bruteforce/bruteforce.h"
#include "grid/counting_sort.h"
#include "grid/find_nbrs.h"
#include "grid/grid.h"
#include "grid/insert_points.h"
// Resampling
#include "grid/find_nbrs_resampling.h"
#include "bruteforce/bruteforce_resampling.h"
#include "grid/find_nbrs_resampling_transform.h"
// Gathering
#include "gather/gather_ppm.h"
#include "gather/gather_cppm.h"
#include "gather/gather_ours_gauss.h"
#include "gather/gather_cuda_table.h"

#include <pybind11/stl.h>

CudaTable& get_global_cuda_table() {
  return GLOBAL_CUDA_TABLE;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("setup_grid_params", &SetupGridParams);

  m.def("insert_points_cuda", &InsertPointsCUDA);
  m.def("test_insert_points_cpu", &TestInsertPointsCPU);

  m.def("counting_sort_cuda", &CountingSortCUDA);
  m.def("counting_sort_cpu", &CountingSortCPU);

  m.def("find_nbrs_cuda", &FindNbrsCUDA);
  m.def("find_nbrs_cpu", &FindNbrsCPU);
  m.def("test_find_nbrs_cpu", &TestFindNbrsCPU);

  // Find nbrs with resampling
  m.def("find_nbrs_resampling_cuda", &FindNbrsResamplingCUDA);
  m.def("find_nbrs_resampling_transform_cuda", &FindNbrsResamplingTransformCUDA);

  // Brute force
  m.def("frnn_bf_cuda", &FRNNBruteForceCUDA);
  m.def("frnn_bf_cpu", &FRNNBruteForceCPU);

  // Brute force resampling
  m.def("frnn_bf_resampling_cuda", &FRNNBruteForceResamplingCUDA);

  // Backprop
  m.def("frnn_backward_cuda", &FRNNBackwardCUDA);

  // Optimized gathering
  m.def("gather_ppm_cuda", &GatherPPMCuda);
  m.def("gather_cppm_cuda", &GatherCPPMCuda);
  m.def("gather_ours_gauss_cuda", &GatherOursGaussCuda);

  // Load the normalization table
  m.def("load_normalization_table", [](const at::Tensor& normalization_table) {
    get_global_cuda_table().load(normalization_table);
  }, "Load the normalization table");
  
}
