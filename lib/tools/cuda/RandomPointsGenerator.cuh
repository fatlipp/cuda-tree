#pragma once

#include <thrust/device_vector.h>

class RPGEngineUniform;

class RandomPointsGenerator
{
  public:
    RandomPointsGenerator(const int3 dims);

  public:
    float2* GenerateOnDevice(const int pointsCount);

  private:
    RPGEngineUniform* rpg;
    thrust::device_vector<float2> points;
};