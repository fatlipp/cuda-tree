#pragma once

class IDrawable
{
public:
    virtual ~IDrawable() = default;

public:
    virtual void Initialize() = 0;
    virtual void Draw() = 0;
};