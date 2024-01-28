#include "tools/cuda/RandomPointsGenerator.cuh"
#include "tools/cuda/CudaHelper.h"

#include <device_launch_parameters.h>
#include <thrust/random.h>

class RPGEngineUniform
{
public:
    RPGEngineUniform(const int3 dims)
        : dims {dims}
        , count {0}
    {
    }
    // Robert Jenkins' 32 bit integer hash function
    __device__ __forceinline__ unsigned hash(unsigned int a) 
    {
      a = (a+0x7ed55d16) + (a<<12);
      a = (a^0xc761c23c) ^ (a>>19);
      a = (a+0x165667b1) + (a<<5);
      a = (a+0xd3a2646c) ^ (a<<9);
      a = (a+0xfd7046c5) + (a<<3);
      a = (a^0xb55a4f09) ^ (a>>16);
      return a;
    }

    __device__ __forceinline__ float2 operator()()
    {
        unsigned seed = hash(blockIdx.x * blockDim.x + threadIdx.x + count);
        count += blockDim.x * gridDim.x;
        thrust::default_random_engine rng(seed);

        thrust::uniform_real_distribution<float> distW(0.02, 0.98f);
        thrust::uniform_real_distribution<float> distH(0.02, 0.98f);
        return float2{ distW(rng) * distW(rng) * dims.x, 
                       distH(rng) * distH(rng) * dims.y };
    }

private:
  const int3 dims;
  int count;
};
RandomPointsGenerator::RandomPointsGenerator(const int3 dims)
{
    rpg = new RPGEngineUniform(dims);
}

float2* RandomPointsGenerator::GenerateOnDevice(const int pointsCount)
{
    points = thrust::device_vector<float2>(pointsCount);
    auto ptr = thrust::raw_pointer_cast(&points[0]);
    
    // memcheck error fix:
    cudaMemset(ptr, 0, points.size() * sizeof(float2));
    // --

    thrust::generate(points.begin(), points.end(), *rpg);
    GET_CUDA_ERROR("generate");

    return ptr;
}