#pragma once

#include "lib/config/BaseConfig.h"

struct RenderConfig : public BaseConfig
{
    int width;
    int height;
    int pointSize;
    int lineWidth;
};