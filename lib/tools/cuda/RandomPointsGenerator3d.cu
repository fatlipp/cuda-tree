#include "tools/cuda/RandomPointsGenerator3d.cuh"
#include "tools/cuda/CudaHelper.h"

#include <device_launch_parameters.h>
#include <thrust/random.h>

class RPGEngineUniform3d
{
public:
    RPGEngineUniform3d(const float3& origin, const float3& dims)
        : origin {origin}
        , dims {dims}
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

    __device__ __forceinline__ float3 operator()()
    {
      unsigned seed = hash(blockIdx.x * blockDim.x + threadIdx.x + count);
      count += blockDim.x * gridDim.x;
      thrust::default_random_engine rng(seed);

      thrust::uniform_real_distribution<float> distW(0.02, 0.98f);
      thrust::uniform_real_distribution<float> distH(0.02, 0.98f);
      thrust::uniform_real_distribution<float> distD(0.02, 0.98f);
      return float3{ distW(rng) * distW(rng) * dims.x, distH(rng) * distH(rng) * dims.y, distD(rng) * distD(rng) * dims.z };
      // return float3{ distW(rng) * dims.x, distH(rng) * dims.y, distD(rng) * dims.z };
    }

private:
  const float3 origin;
  const float3 dims;
  int count;
};

RandomPointsGenerator3d::RandomPointsGenerator3d(const float3& origin, const float3& dims)
{
    rpg = new RPGEngineUniform3d(origin, dims);
}

float3* RandomPointsGenerator3d::GenerateOnDevice(const int pointsCount)
{
    points = thrust::device_vector<float3>(pointsCount);
    auto ptr = thrust::raw_pointer_cast(&points[0]);
    
    // memcheck error fix:
    cudaMemset(ptr, 0, points.size() * sizeof(float3));
    // --

    thrust::generate(points.begin(), points.end(), *rpg);
    GET_CUDA_ERROR("generate");

    return ptr;
}