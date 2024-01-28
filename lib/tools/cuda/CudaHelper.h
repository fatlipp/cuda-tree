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
  cudaDeviceSynchronize();
}