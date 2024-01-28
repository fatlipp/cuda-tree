#pragma once

#include <vector>

template<typename T, typename P>
class ITreeBuilder
{

public:
    virtual ~ITreeBuilder() = default;

public:
    virtual void Initialize(const int capacity) = 0;
    virtual void Build(P* points, const int count) = 0;
    virtual void Reset() = 0;

    const T& GetTree() const
    {
        return *tree;
    }

protected:
    T* tree;
};