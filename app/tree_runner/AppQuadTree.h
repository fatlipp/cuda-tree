#include "lib/core/TreeExplorerCpu.h"
#include "lib/tools/cuda/CudaHelper.h"
#include "lib/tools/cuda/RandomPointsGenerator.cuh"
#include "lib/core/quad_tree/QuadTreeBuilderCuda.cuh"
#include "render/base/RenderCamera.h"
#include "drawable/DrawableTree.h"
#include "render/tools/ScreenClick2d.h"

#include <iostream>

void RunAppQuadTree(const TreeConfig& treeConfig, const RenderConfig& renderConfig, const AppConfig& appConfig)
{
    std::cout << "RunAppQuadTree()\n";

    RandomPointsGenerator generator(treeConfig.origin, treeConfig.size);
    float2* points = generator.GenerateOnDevice(appConfig.pointsCount);

    auto treeBuilder = std::make_unique<QuadTreeBuilderCuda>(treeConfig);
    treeBuilder->Initialize(appConfig.pointsCount);
    treeBuilder->Build(points, appConfig.pointsCount);

    const auto& tree = treeBuilder->GetTree();

    const int maxNodes = treeConfig.GetNodesCount();

    // QuadTree* treeHost;
    // deviceToHost(&tree, maxNodes, &treeHost);

    float2* pointsHost;
    deviceToHost(points, appConfig.pointsCount, &pointsHost);

    if (!appConfig.enableRender)
    {
        std::cout << "SOME TEST: " << std::endl;

        std::cout << "1. kNN:" << std::endl;
        std::vector<Neighbour> result;
        result = NearestNeighbours<float2, 2>(&tree, treeConfig, {0.0f, 0.0f}, 15, pointsHost);
        std::cout << "Items found: " << result.size() << std::endl << std::endl;

        std::cout << "2. RadiusSearch:" << std::endl;
        result = RadiusSearch<float2, 2>(&tree, treeConfig, {0.0f, 0.0f}, 15, pointsHost);
        std::cout << "Items found: " << result.size() << std::endl;

        return;
    }

    auto drawableTree = std::make_unique<DrawableTree<float2, 2>>(&tree, pointsHost, treeConfig, 
        renderConfig.pointSize, renderConfig.lineWidth);

    RenderCamera render(renderConfig, false);
    render.AddDrawable(drawableTree.get());

    ScreenClick2d clicker(render, tree.bounds.min, tree.bounds.max);
    render.AddMouseClickCallback([&clicker, &tree, &treeConfig, &pointsHost](const int x, const int y, const int btn) {
            float2 outPos;
            if (clicker(x, y, outPos))
            {
                std::vector<Neighbour> result;

                switch (btn)
                {
                case 1:
                    result = NearestNeighbours<float2, 2>(&tree, treeConfig, outPos, 5, pointsHost);
                    break;
                case 2:
                    result = RadiusSearch<float2, 2>(&tree, treeConfig, outPos, 1.25f, pointsHost);
                    break;
                
                default:
                    break;
                }
                std::cout << "result: " << result.size() << std::endl;
            }
        });
    render.Initialize();
    render.StartLoop();

    free(pointsHost);
}