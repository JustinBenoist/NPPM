#include <tuple>

#include "grid/counting_sort.h"
#include "grid/grid.h"
#include "utils/mink.cuh"
// customized dispatch utils for our function type
#include "utils/dispatch.h"
#include "utils/bsdf.h"

// We use https://github.com/bacTlink/mitsuba-CPPM/blob/master/src/integrators/cppm/cppm3.cpp

#define MIN_PHOTON_COUNT 10.f
#define SEC_U 2
#define SEC_V 6
#define BETA 1.2

__host__ __device__ inline float Dot(const float* a, const float* b){
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__host__ __device__ inline int getSection(const float photonPos_x,
                                          const float photonPos_y,
                                          const float photonPos_z,
                                          const float &radius) {
    float angle = atan2(photonPos_y, photonPos_x);
    angle += photonPos_y < 0 || (photonPos_y == 0 && photonPos_x < 0) ? 2 * M_PI : 0;
    float dot = photonPos_x * photonPos_x + photonPos_y * photonPos_y + photonPos_z * photonPos_z;
    int i = std::max(0, std::min(SEC_U - 1, (int)(SEC_U * dot / (radius * radius))));
    int j = (int)(SEC_V * angle / (M_PI * 2)) % SEC_V;
    return i * SEC_V + j;
}

__host__ __device__ inline void chiSquaredTest(int32_t *photonCount, bool* passed) {
    const float chi2_95[] = {100, 3.84146, 5.99146, 7.81473, 9.48773, 11.07050, 12.59159, 14.06714, 15.50731, 16.91898, 18.30704, 19.67514, 21.02607, 22.36203, 23.68479, 24.99579, 26.29623, 27.58711, 28.86930, 30.14353, 31.41043, 32.67057, 33.92444, 35.17246, 36.41503, 37.65248, 38.88514, 40.11327, 41.33714, 42.55697, 43.77297, 44.98534, 46.19426, 47.39988, 48.60237, 49.80185, 50.99846, 52.19232, 53.38354, 54.57223, 55.75848, 56.94239, 58.12404, 59.30351, 60.48089, 61.65623, 62.82962, 64.00111, 65.17077, 66.33865};

    float sumO = 0.f, sumSqrO = 0.f, V = 0.f;
    for (int i = 0; i < SEC_U; ++i) {
        for (int j = 0; j < SEC_V; ++j) {
            int32_t O = photonCount[i * SEC_V + j];
            sumO += O;
            sumSqrO += O * O;
        }
        int secs = (i + 1) * SEC_V;
        if (sumO == 0)
            passed[i] = true;
        else {
            V = sumSqrO * secs / (double)sumO - sumO;
            passed[i] = V < chi2_95[secs - 1];
        }
    }
}

// TODO: add docs
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
);
