#pragma once

#include "lib/config/ConfigParserBase.h"
#include "lib/config/TreeConfig.h"

NLOHMANN_JSON_SERIALIZE_ENUM(TreeType, {
    {TreeType::QUADTREE, "Quad"},
    {TreeType::OCTREE, "Oct"},
})

namespace config
{

TreeConfig ParseTreeConfig(const std::string& path)
{
    const nlohmann::json data = ParseBase(path);

    if (data.empty()) { return {}; }
    
    TreeConfig result;
    result.isValid = true;
    result.type = data["type"];

    result.threadsPerBlock = data["threadsPerBlock"];
    result.maxDepth = data["maxDepth"];
    result.minPointsToDivide = data["minPointsToDivide"];
        
    result.origin.x = data["origin"]["x"];
    result.origin.y = data["origin"]["y"];
    result.origin.z = data["origin"]["z"];
    result.size.x = data["size"]["x"];
    result.size.y = data["size"]["y"];
    result.size.z = data["size"]["z"];

    return result;
}
    
} // namespace config

