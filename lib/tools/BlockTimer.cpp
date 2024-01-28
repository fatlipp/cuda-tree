#include "tools/BlockTimer.h"

void BlockTimer::Start()
{
    this->startTime = std::chrono::high_resolution_clock::now();
}

void BlockTimer::Stop()
{
    const auto endTime = std::chrono::high_resolution_clock::now();
    lastTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
}