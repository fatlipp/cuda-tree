#pragma once

#include <vector_types.h>

// 2d:
inline  __host__ __device__ float norm(float2 a)
{
    return a.x * a.x + a.y * a.y;
}

inline  __host__ __device__ bool operator<(float2 a, float2 b)
{
    return a.x < b.x && a.y < b.y;
}
inline  __host__ __device__ bool operator>=(float2 a, float2 b)
{
    return a.x >= b.x && a.y >= b.y;
}

inline  __host__ __device__ float2 operator*(float2 a, float b)
{
    return { a.x * b, a.y * b};
}

inline  __host__ __device__ float2 operator*(float2 a, float2 b)
{
    return { a.x * b.x, a.y * b.y};
}

inline  __host__ __device__ void operator*=(float2 &a, float2 b)
{
    a.x *= b.x;
    a.y *= b.y;
} 

inline  __host__ __device__ float2 operator/(float2 a, float2 b)
{
    return { a.x / b.x, a.y / b.y };
}

inline  __host__ __device__ void operator/=(float2 &a, float2 b)
{
    a.x /= b.x;
    a.y /= b.y;
} 

inline  __host__ __device__ float2 operator/(float2 a, float b)
{
    return { a.x / b, a.y / b };
}

inline  __host__ __device__ float2 operator+(float2 a, float2 b)
{
    return { a.x + b.x, a.y + b.y };
}

inline  __host__ __device__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
} 

inline  __host__ __device__ float2 operator-(float2 a, float2 b)
{
    return { a.x - b.x, a.y - b.y };
}

inline  __host__ __device__ void operator-=(float2 &a, float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
} 

// 3d:
inline  __host__ __device__ float norm(float3 a)
{
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

inline  __host__ __device__ bool operator<(float3 a, float3 b)
{
    return a.x < b.x && a.y < b.y && a.z < b.z;
}
inline  __host__ __device__ bool operator>=(float3 a, float3 b)
{
    return a.x >= b.x && a.y >= b.y && a.z >= b.z;
}

inline  __host__ __device__ float3 operator*(float3 a, float b)
{
    return { a.x * b, a.y * b, a.z * b };
}

inline  __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return { a.x * b.x, a.y * b.y, a.z * b.z };
}

inline  __host__ __device__ void operator*=(float3 &a, float3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
} 

inline  __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return { a.x / b.x, a.y / b.y, a.z / b.z };
}

inline  __host__ __device__ void operator/=(float3 &a, float3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
} 

inline  __host__ __device__ float3 operator/(float3 a, float b)
{
    return { a.x / b, a.y / b, a.z / b};
}

inline  __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

inline  __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
} 

inline  __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

inline  __host__ __device__ void operator-=(float3 &a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
} 