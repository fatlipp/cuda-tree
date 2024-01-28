#include <cstring>
#include <iostream>
#include <cuda_gl_interop.h>

void getLastCudaError(const char *errorMessage, const char *file,
                               const int line) 
{                               
  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) 
  {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}