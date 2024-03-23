#pragma once

#include "lib/config/BaseConfig.h"

#include <string>

struct AppConfig : public BaseConfig
{
    int pointsCount = 1000000;
    std::string treeConfigPath;
    std::string renderConfigPath;
    std::string modelPath;
    
    bool enableRender;
};