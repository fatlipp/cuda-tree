#pragma once

#include <chrono>

class BlockTimer
{
public:
    BlockTimer() 
        : lastTime {0}
    {}

public:
    void Start();
    void Stop();

    float Get() { return lastTime; }

private:
    std::chrono::high_resolution_clock::time_point startTime;
    float lastTime;
};
