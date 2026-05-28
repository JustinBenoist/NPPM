#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

// Information about the table used for normalization
#define NORMALIZATION_TABLE_SIZE 1024
#define TABLE_MIN logf(0.0001)
#define TABLE_MAX logf(50.0)

// Cuda texture struct 
struct CudaTable {
    cudaTextureObject_t tex = 0;
    cudaArray_t arr = nullptr;
  
    void load(const at::Tensor& normalization_table) {
      TORCH_CHECK(normalization_table.is_cuda(), "Tensor must be on CUDA");
      TORCH_CHECK(normalization_table.dtype() == at::kFloat, "Tensor must be float32");
      // TORCH_CHECK(normalization_table.dim() == 2, "Tensor must be 2D");
  
      std::cout << "Load normalization table \n";
      at::Tensor tensor_contig = normalization_table.contiguous();

      // Local object to hold the texture object and array
      cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  
      cudaMallocArray(&arr, &channelDesc, NORMALIZATION_TABLE_SIZE, NORMALIZATION_TABLE_SIZE);

      cudaMemcpyKind kind = tensor_contig.is_cuda() ?
                            cudaMemcpyDeviceToDevice :
                            cudaMemcpyHostToDevice;
      // Use flat copy for simplicity
      cudaMemcpyToArray(
          arr, 0, 0,
          tensor_contig.data_ptr<float>(),
          sizeof(float) * NORMALIZATION_TABLE_SIZE * NORMALIZATION_TABLE_SIZE,
          kind);
  
      cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      resDesc.resType = cudaResourceTypeArray;
      resDesc.res.array.array = arr;
  
      cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      texDesc.addressMode[0] = cudaAddressModeClamp;
      texDesc.addressMode[1] = cudaAddressModeClamp;
      texDesc.filterMode = cudaFilterModePoint;
      texDesc.readMode = cudaReadModeElementType;
      texDesc.normalizedCoords = true;
  
      AT_CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));

      // float* debug_host = new float[NORMALIZATION_TABLE_SIZE * NORMALIZATION_TABLE_SIZE];
      // cudaMemcpyFromArray(debug_host, arr, 0, 0,
      //     sizeof(float) * NORMALIZATION_TABLE_SIZE * NORMALIZATION_TABLE_SIZE,
      //     cudaMemcpyDeviceToHost);

      // std::cout << "Center value: " << debug_host[511 * NORMALIZATION_TABLE_SIZE + 511] << "\n";
    }
  
    ~CudaTable() {
      // std::cout << "Destroy\n";
      if (tex) cudaDestroyTextureObject(tex);
      if (arr) cudaFreeArray(arr);
    }
  };

// Define the static global instance here
extern CudaTable GLOBAL_CUDA_TABLE;