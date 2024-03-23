#include "lib/config/AppConfigParser.h"
#include "lib/config/TreeConfigParser.h"
#include "lib/config/RenderConfigParser.h"

#include "tree_runner/AppOctree.h"
#include "tree_runner/AppOctreeObj.h"
#include "tree_runner/AppQuadTree.h"

#include <iostream>

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Please add a patch to a config file" << std::endl;

        return -1;
    }

    auto appConfig = config::ParseAppConfig(argv[1]);
    if (!appConfig.isValid) return 1;

    auto treeConfig = config::ParseTreeConfig(appConfig.treeConfigPath);
    if (!treeConfig.isValid) return 1;

    const auto renderConfig = (!appConfig.renderConfigPath.empty()) ? 
        config::ParseRenderConfig(appConfig.renderConfigPath) : RenderConfig{true, 2000, 2000, false};
    if (!renderConfig.isValid) return 1;

    if (treeConfig.maxDepth > 7)
    {
        std::cout << "Current implementation doesn't work well when 'maxDepth > 7'" << std::endl;
        std::cout << 
            "Remove the line below and recompile to remove the restriction (tune tree configs to achieve a stable result)" 
            << std::endl;
        treeConfig.maxDepth = 7;
    }

    if (treeConfig.type == TreeType::OCTREE)
    {
        RunAppOctreeObj(treeConfig, renderConfig, appConfig);
    }
    else
    {
        std::cerr << "A current implementation doesn't support 2d models (Use 'Octree' mode).\n";
    }

    return 0;
}