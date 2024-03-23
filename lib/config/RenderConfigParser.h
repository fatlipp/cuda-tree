#pragma once

#include "lib/config/ConfigParserBase.h"
#include "lib/config/RenderConfig.h"

namespace config
{

RenderConfig ParseRenderConfig(const std::string& path)
{
    const nlohmann::json data = ParseBase(path);

    if (data.empty()) { return {}; }
    
    RenderConfig result;
    result.isValid = true;
    result.width = data["width"];
    result.height = data["height"];
    result.pointSize = data["point_size"];
    result.lineWidth = data["line_width"];
    result.cameraSpeed = data["camera_speed"];
    result.mouseSpeed = data["mouse_speed"];

    return result;
}
    
} // namespace config

