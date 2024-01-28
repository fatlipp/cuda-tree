#pragma once

#include "lib/core/CudaOperator.cuh"

template<typename T>
struct AABB
{
    T min;
    T max;

    __host__ __device__ bool Check(const T& point) const
    {
        return point >= min && point < max; 
    }

    __host__ __device__ T GetCenter() const
    {
        return (min + max) * 0.5f;
    }

    __host__ __device__ T Size() const
    {
        return max - min;
    }
};