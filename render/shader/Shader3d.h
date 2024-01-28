#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "tools/OpenGlHelper.h"

static const char* vertexShaderSource = R"(
    #version 330 core

    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aCol;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    out vec3 v_color; 

    void main() {
        gl_Position = projection * view * model * vec4(-aPos.x, aPos.y, aPos.z, 1.0);
        v_color = aCol;
    }
)";

static const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    in vec3 v_color;
    void main() {
        FragColor = vec4(v_color, 1.0);
    }
)";

static GLuint CreateDefaultShader3d()
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    GET_GL_ERROR("CreateDefaultShader3d");

    return shaderProgram;
}