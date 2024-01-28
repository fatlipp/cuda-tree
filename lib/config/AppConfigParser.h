#pragma once

#include "lib/config/ConfigParserBase.h"
#include "lib/config/AppConfig.h"

#include <regex>

namespace config
{
void SolveRelativePath(std::string& path, const std::string& parentPath)
{
    std::regex configNameRegex("\\./.+.json");
    std::smatch relativePathMatch;

    if (std::regex_search(path, relativePathMatch, configNameRegex) &&
        relativePathMatch.prefix().length() == 0)
    {
        std::regex parentFolderRegex("[a-zA-Z0-9]+.json");
        std::smatch parentFolderMatch;
        if (std::regex_search(parentPath, parentFolderMatch, parentFolderRegex) &&
            parentFolderMatch.prefix().length() > 0)
        {
            path = parentFolderMatch.prefix().str() + path.substr(2);
        }
    }
}

AppConfig ParseAppConfig(const std::string& path)
{
    const nlohmann::json data = ParseBase(path);

    if (data.empty()) { return {}; }

    AppConfig result;
    result.isValid = true;
    result.pointsCount = data["points_count"];
    result.treeConfigPath = data["tree_config"];
    result.renderConfigPath = data["render_config"];
    result.enableRender = data["enable_render"];

    SolveRelativePath(result.treeConfigPath, path);
    SolveRelativePath(result.renderConfigPath, path);

    return result;
}
    
} // namespace config

