#pragma once

#include "vector_types.h"

template<int DIMENSION>
__host__ __device__ static int GetNodeByDepth(const int depth)
{
    return 0;
}

template<>
__host__ __device__ int GetNodeByDepth<2>(const int depth)
{
    int sum = 1;
    for (int i = 0; i < depth; ++i)
    {
        sum *= 4;
    }

    return sum;
}

template<>
__host__ __device__ int GetNodeByDepth<3>(const int depth)
{
    int sum = 1;
    for (int i = 0; i < depth; ++i)
    {
        sum *= 8;
    }

    return sum;
}