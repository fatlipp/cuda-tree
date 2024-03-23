#pragma once

#include <thrust/device_vector.h>

class RPGEngineUniform3d;

class RandomPointsGenerator3d
{
  public:
    RandomPointsGenerator3d(const float3& origin, const float3& dims);

  public:
    float3* GenerateOnDevice(const int pointsCount);

  private:
    RPGEngineUniform3d* rpg;
    thrust::device_vector<float3> points;
};