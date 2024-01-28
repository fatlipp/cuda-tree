#pragma once

#include "render/base/IDrawable.h"
#include "lib/config/RenderConfig.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <functional>
#include <vector>
#include <memory>

class Render
{
public:
    Render() 
        : config{} 
        {}

    Render(const RenderConfig& config);
    virtual ~Render();

public:
    // drawables:
    void AddDrawable(IDrawable* drawable)
    {
        drawables.push_back(drawable);
    }

    // callbacks:
    void AddMouseClickCallback(
        std::function<void(const int, const int, const int)>&& cb)
    {
        mouseClickCallback.push_back(std::move(cb));
    }
    void AddMouseReleaseCallback(
        std::function<void(const int, const int, const int)>&& cb)
    {
        mouseReleaseCallback.push_back(std::move(cb));
    }
    void AddMouseMoveCallback(
        std::function<void(const int, const int)>&& cb)
    {
        mouseMoveCallback.push_back(std::move(cb));
    }
    void AddKeyboardCallback(
        std::function<void(const int, const int)>&& cb)
    {
        keyboardCallback.push_back(std::move(cb));
    }

public:
    virtual void Initialize();
    virtual void OnDisplayStart() { }
    virtual void OnDisplayEnd() { }

    void StartLoop();

public:
    std::pair<int, int> GetSize() const
    {
        return {config.width, config.height};
    }

    std::pair<int, int> GetMousePos() const
    {
        return {mouseX, mouseY};
    }

protected:
    const RenderConfig config;

    std::vector<IDrawable*> drawables;

    GLFWwindow* window;
    std::vector<std::function<void(const int, const int, const int)>> mouseClickCallback;
    std::vector<std::function<void(const int, const int, const int)>> mouseReleaseCallback;
    std::vector<std::function<void(const int, const int)>> mouseMoveCallback;
    std::vector<std::function<void(const int, const int)>> keyboardCallback;

    int mouseX;
    int mouseY;
};