#include "base/Render.h"
#include "shader/Shader3d.h"
#include "base/RenderCamera.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

void RenderCamera::Initialize()
{
    Render::Initialize();

    shaderProgram = CreateDefaultShader3d();

    buttonState = ButtonState::NONE;
    oldX = -1;
    oldY = -1;

    AddMouseClickCallback([&](const int x, const int y, const int c) {
        if (c == 1) 
        {
            buttonState = ButtonState::ROTATE;
        }
        else if (c == 2) 
        {
            buttonState = ButtonState::SCALE;
        }
        oldX = x;
        oldY = y;
        dx = 0;
        dy = 0;
    });
    AddMouseReleaseCallback([&](const int x, const int y, const int c) {
        if (buttonState == ButtonState::ROTATE && c == 1)
        {
            buttonState = ButtonState::NONE;
        }
    });
    AddMouseMoveCallback([&](const int x, const int y) {
        dx = (float)(x - oldX);
        dy = (float)(y - oldY);
        oldX = x;
        oldY = y;
    });
    AddKeyboardCallback([&](const int key, const int action) {

            if (action == GLFW_PRESS || action == GLFW_REPEAT)
            {
                switch (key)
                {
                case GLFW_KEY_W:
                    buttonState = ButtonState::MOVE_FWD;
                    break;
                case GLFW_KEY_S:
                    buttonState = ButtonState::MOVE_BCK;
                    break;
                case GLFW_KEY_A:
                    buttonState = ButtonState::MOVE_LEFT;
                    break;
                case GLFW_KEY_D:
                    buttonState = ButtonState::MOVE_RIGHT;
                    break;
                case GLFW_KEY_R:
                    buttonState = ButtonState::MOVE_UP;
                    break;
                case GLFW_KEY_F:
                    buttonState = ButtonState::MOVE_DOWN;
                    break;
                
                default:
                    break;
                }
            }
            else
            {
                buttonState = ButtonState::NONE;
            }
        });
}

void RenderCamera::OnDisplayStart()
{
    glEnable(GL_DEPTH_TEST);

    const glm::vec3 direction(
            cos(verticalAngle) * sin(horizontalAngle),
            sin(verticalAngle),
            cos(verticalAngle) * cos(horizontalAngle)
        );
    const glm::vec3 right = glm::vec3(
            sin(horizontalAngle - 3.14f / 2.0f),
            0,
            cos(horizontalAngle - 3.14f / 2.0f)
        );
    const glm::vec3 up = glm::cross(right, direction );

    switch (buttonState)
    {
    case ButtonState::ROTATE:
        if (is3d)
        {
            horizontalAngle += mouseSpeed * dx;
            verticalAngle   += mouseSpeed * dy;
        }
        break;
    case ButtonState::MOVE_FWD:
        position += speed * direction;
        break;
    case ButtonState::MOVE_BCK:
        position -= speed * direction;
        break;
    case ButtonState::MOVE_LEFT:
        position -= speed * right;
        break;
    case ButtonState::MOVE_RIGHT:
        position += speed * right;
        break;
    case ButtonState::MOVE_UP:
        position += speed * up;
        break;
    case ButtonState::MOVE_DOWN:
        position -= speed * up;
        break;
    default:
        break;
    }

    projectionMatrix = glm::perspective(glm::radians(60.0f), static_cast<float>(config.width) / config.height, 0.01f, 10000.0f);
    viewMatrix = glm::lookAt(position, position + direction, up);

    glUseProgram(shaderProgram);
    
    const GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");
    const GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");
    const GLuint projectionLoc = glGetUniformLocation(shaderProgram, "projection");

    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(modelMatrix));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projectionMatrix));
}

void RenderCamera::OnDisplayEnd()
{
    glUseProgram(0);
}

RenderCamera::~RenderCamera()
{
    glDeleteProgram(shaderProgram);
}