#include "lib/core/quad_tree/QuadTreeBuilderCuda.cuh"
#include "lib/core/quad_tree/QuadTreeKernel.cuh"
#include "lib/tools/cuda/CudaHelper.h"

#include <cuda_runtime.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <set>
#include <sstream>
#include <algorithm>

QuadTreeBuilderCuda::QuadTreeBuilderCuda(const TreeConfig& config)
    : config {config}
{
}

void QuadTreeBuilderCuda::Initialize(const int capacity)
{
    cudaMalloc((void**)&pointsExch, capacity * sizeof(float2));
    GET_CUDA_ERROR("cudaMalloc() pointsExch");
    // tree:
    const int maxNodes = config.GetNodesCount();
    cudaMallocManaged((void**)&tree, maxNodes * sizeof(QuadTree));
    GET_CUDA_ERROR("cudaMallocManaged() tree");
    cudaDeviceSynchronize();
    GET_CUDA_ERROR("cudaDeviceSynchronize");
}

void QuadTreeBuilderCuda::Build(float2* points, const int size)
{
    Reset();

    tree->id = 0;
    tree->bounds.min = {config.origin.x, config.origin.y};
    const auto maxDim = config.origin + config.size;
    tree->bounds.max = {maxDim.x, maxDim.y};
    tree->startId = 0;
    tree->endId = size;

    std::cout << "Build()" << std::endl;

    auto satrtTime = std::chrono::high_resolution_clock::now();
    const int warpsPerBlock = config.threadsPerBlock / 32;

    cudaMemcpy(pointsExch, points, size * sizeof(float2), cudaMemcpyDeviceToDevice);

    QuadTreeKernel<<<1, config.threadsPerBlock, warpsPerBlock * 4 * sizeof(int)>>>(
        points, pointsExch, tree, 
        0, config.maxDepth, config.minPointsToDivide);
    GET_CUDA_ERROR("KernelError");
    cudaDeviceSynchronize();
    GET_CUDA_ERROR("SyncError");
    auto endTime = std::chrono::high_resolution_clock::now();
    std::cout << "Kernel() duration: " << 
        (std::chrono::duration_cast<std::chrono::milliseconds>(endTime - satrtTime).count()) << std::endl;

    QuadTree* tree2 = tree;
    int totalCount = 0;

    for (int depth = 0; depth < config.maxDepth; ++depth)
    {
        const auto leafs = GetNodeByDepth<2>(depth);
        for (int leaf = 0; leaf < leafs; ++leaf)
        {
            const QuadTree* const subTree = &tree2[leaf];

            if ((subTree->PointsCount() < config.minPointsToDivide || 
                depth == config.maxDepth - 1) && subTree->PointsCount() > 0)
            {
                totalCount += subTree->PointsCount();
            }
        }

        tree2 += leafs;
    }

    std::cout << "total points: " << totalCount << " / " << size << "\n";

    if (totalCount != size)
    {
        throw "Invalid tree: totalCount != size\n";
    }
}

void QuadTreeBuilderCuda::Reset()
{
    std::cout << "Reset()" << std::endl;
    const int maxNodes = config.GetNodesCount();

    for (int i = 0; i < maxNodes; ++i)
    {
        tree[i].id = 0;
        tree[i].bounds.min = {0.0f, 0.0f};
        tree[i].bounds.max = {0.0f, 0.0f};
        tree[i].startId = 0;
        tree[i].endId = 0;
    }
}

QuadTreeBuilderCuda::~QuadTreeBuilderCuda()
{
    cudaFree(pointsExch);
    cudaFree(tree);
}
