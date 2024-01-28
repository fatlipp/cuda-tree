#pragma once

#include "lib/core/ITreeBuilder.h"
#include "lib/core/quad_tree/QuadTree.h"
#include "lib/config/TreeConfig.h"
#include "lib/tools/cuda/CudaHelper.h"

class QuadTreeBuilderCuda : public ITreeBuilder<QuadTree, float2>
{
public:
    QuadTreeBuilderCuda(const TreeConfig& config);
    ~QuadTreeBuilderCuda() override;

public:
    void Initialize(const int capacity) override;
    void Build(float2* point, const int size) override;
    void Reset() override;

private:
    float2* pointsExch;
    TreeConfig config;
};