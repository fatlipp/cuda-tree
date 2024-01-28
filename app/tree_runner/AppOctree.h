#include "lib/core/TreeExplorerCpu.h"
#include "lib/tools/cuda/RandomPointsGenerator3d.cuh"
#include "render/base/RenderCamera.h"
#include "drawable/DrawableTree.h"
#include "lib/core/octree/OctreeBuilderCuda.cuh"

#include <iostream>

void RunAppOctree(const TreeConfig& treeConfig, const RenderConfig& renderConfig, const AppConfig& appConfig)
{
    std::cout << "RunAppOctree()\n";
    
    RandomPointsGenerator3d generator(treeConfig.size);
    float3* points = generator.GenerateOnDevice(appConfig.pointsCount);

    auto treeBuilder = std::make_unique<OctreeBuilderCuda>(treeConfig);
    treeBuilder->Initialize(appConfig.pointsCount);
    treeBuilder->Build(points, appConfig.pointsCount);

    const auto& tree = treeBuilder->GetTree();

    float3* pointsHost;
    deviceToHost(points, appConfig.pointsCount, &pointsHost);

    if (!appConfig.enableRender)
    {
        std::cout << "SOME TEST: " << std::endl;

        std::cout << "1. kNN:" << std::endl;
        std::vector<Neighbour> result;
        result = NearestNeighbours<float3, 3>(&tree, treeConfig, {0.0f, 0.0f, 0.0f}, 15, pointsHost);
        std::cout << "Items found: " << result.size() << std::endl << std::endl;

        std::cout << "2. RadiusSearch:" << std::endl;
        result = RadiusSearch<float3, 3>(&tree, treeConfig, {0.0f, 0.0f, 0.0f}, 15, pointsHost);
        std::cout << "Items found: " << result.size() << std::endl;

        return;
    }

    RenderCamera render(renderConfig, pointsHost, true);

    auto drawableTree = std::make_unique<DrawableTree<float3, 3>>(&tree, pointsHost, treeConfig, 
        renderConfig.pointSize, renderConfig.lineWidth);
    render.AddDrawable(drawableTree.get());
    
    render.Initialize();
    render.StartLoop();
}