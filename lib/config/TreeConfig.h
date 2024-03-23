#pragma once

#include "lib/config/BaseConfig.h"
#include "lib/core/TreeType.h"

#include <vector_types.h>

struct TreeConfig : public BaseConfig
{
    TreeType type;
    float3 origin;
    float3 size;
    int maxDepth;
    
    int threadsPerBlock;
    int minPointsToDivide;

    inline int GetNodesCount() const
    {
        int maxNodes = 0;
        for (int i = 0; i < maxDepth; ++i)
        {
            maxNodes += std::pow(type == TreeType::QUADTREE ? 4 : 8, i);
        }

        return maxNodes;
    }
};