#pragma once

#include <iostream>
#include <vector_types.h>

void getLastCudaError(const char *errorMessage, const char *file, const int line);
#define GET_CUDA_ERROR(msg) getLastCudaError(msg, __FILE__, __LINE__);

template<typename T>
void deviceToHost(const T* devPtr, const int size, T** hostPtr)
{
  *hostPtr = (T*)malloc(size * sizeof(T));

  cudaMemcpy(*hostPtr, devPtr, size * sizeof(T), cudaMemcpyDeviceToHost);
  GET_CUDA_ERROR("cudaMemcpy() deviceToHost");
  cudaDeviceSynchronize();
  GET_CUDA_ERROR("cudaDeviceSynchronize() deviceToHost");
}
template<typename T>
void hostToDevice(const T* hostPtr, const int size, T** devPtr)
{
  cudaMalloc((void**)devPtr, size * sizeof(T));

  cudaMemcpy(*devPtr, hostPtr, size * sizeof(T), cudaMemcpyHostToDevice);
  GET_CUDA_ERROR("cudaMemcpy() hostToDevice");
  cudaDeviceSynchronize();
  GET_CUDA_ERROR("cudaDeviceSynchronize() hostToDevice");
}