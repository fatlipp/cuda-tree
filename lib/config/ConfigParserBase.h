#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <thirdparty/nlohmann/json.hpp>

namespace config
{

nlohmann::json ParseBase(const std::string& path)
{
    nlohmann::json config;
    std::ifstream inpStream(path);
    if (!inpStream.is_open())
    {
        std::cerr << "file not found: " << path << std::endl;
        return {};
    }

    inpStream >> config;
    inpStream.close();

    return config;
}
    
} // namespace config