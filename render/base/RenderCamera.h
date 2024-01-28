#pragma once

#include "lib/config/RenderConfig.h"
#include "render/base/Render.h"

#include <glm/glm.hpp>

enum class ButtonState
{
    NONE,
    ROTATE,
    SCALE,
    MOVE_FWD,
    MOVE_BCK,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_UP,
    MOVE_DOWN
};

class RenderCamera : public Render
{
public:
    RenderCamera(const RenderConfig& config, const void* data, const bool is3d)
        : Render(config)
        , data {data}
        , is3d {is3d}
    {
    }

    ~RenderCamera() override;

public:
    void Initialize() override;
    void OnDisplayStart() override;
    void OnDisplayEnd() override;

public:
    glm::vec3 GetPosition() const
    {
        return position;
    }

    glm::mat4 GetModelMatrix() const
    {
        return modelMatrix;
    }

    glm::mat4 GetViewMatrix() const
    {
        return viewMatrix;
    }

    glm::mat4 GetProjectionMatrix() const
    {
        return projectionMatrix;
    }

private:
    const void* data;
    const bool is3d;

    GLuint shaderProgram;

private:
    ButtonState buttonState;
    int oldX;
    int oldY;
    float dx;
    float dy;

    // position
    float horizontalAngle = 0.0f;
    float verticalAngle = 0.0f;

    float speed = 2.0f;
    float mouseSpeed = 0.001f;

    glm::vec3 position = glm::vec3( 0, 0, -5 );
    glm::mat4 modelMatrix = glm::mat4(1.0);
    glm::mat4 viewMatrix = glm::mat4(1.0);
    glm::mat4 projectionMatrix = glm::mat4(1.0);

};