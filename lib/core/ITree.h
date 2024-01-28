#pragma once

#include "lib/core/AABB.h"

#include <iostream>

template<typename T>
struct ITree
{
    int id;
    int startId;
    int endId;
    AABB<T> bounds;

    __host__ __device__ bool Check(const T& point) const
    {
        return bounds.Check(point);
    }

    __host__ __device__ int PointsCount() const
    {
        return endId - startId;
    }
};