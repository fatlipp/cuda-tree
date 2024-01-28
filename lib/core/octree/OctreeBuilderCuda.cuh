#pragma once

#include "lib/core/ITreeBuilder.h"
#include "lib/core/octree/Octree.h"
#include "lib/config/TreeConfig.h"
#include "lib/tools/cuda/CudaHelper.h"

class OctreeBuilderCuda : public ITreeBuilder<Octree, float3>
{
public:
    OctreeBuilderCuda(const TreeConfig& config);
    ~OctreeBuilderCuda() override;

public:
    void Initialize(const int capacity) override;
    void Build(float3* point, const int count) override;
    void Reset() override;

private:
    float3* pointsExch;
    TreeConfig config;
};