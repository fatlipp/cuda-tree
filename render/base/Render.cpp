#include "base/Render.h"
#include "tools/OpenGlHelper.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>

Render::Render(const RenderConfig& config)
  : config { config }
{
}

void Render::Initialize()
{
  if (!glfwInit()) 
  {
    fprintf(stderr, "Failed to GLFW init\n");
    return;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window = glfwCreateWindow(config.width, config.height, "Render", NULL, NULL);
  if (!window) 
  {
    fprintf(stderr, "Failed to create a Window\n");
    glfwTerminate();
    return;
  }

  glfwMakeContextCurrent(window);

  if (glewInit() != GLEW_OK) 
  {
      fprintf(stderr, "Failed to initialize GLEW\n");
      return;
  }

  GLenum error = glGetError();
  if (error != GL_NO_ERROR) 
  {
      return;
  }

  glfwSetWindowUserPointer(window, this);

  glfwSetCursorPosCallback(window, [](GLFWwindow* window, double x, double y) {
      auto render = static_cast<Render*>(glfwGetWindowUserPointer(window));
      render->mouseX = x;
      render->mouseY = y;

      for (auto& c : render->mouseMoveCallback)
        c(render->mouseX, render->mouseY);
    });
  glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int button, int action, int mods) {

      auto render = static_cast<Render*>(glfwGetWindowUserPointer(window));

      if (action == GLFW_PRESS)
      {
        const int btnId = button == GLFW_MOUSE_BUTTON_LEFT ? 1 : button == GLFW_MOUSE_BUTTON_RIGHT ? 2 : 
          button == GLFW_MOUSE_BUTTON_MIDDLE ? 3 : 0;

        for (auto& c : render->mouseClickCallback)
          c(render->mouseX, render->mouseY, btnId);
      }
      else if (action == GLFW_RELEASE)
      {
        const int btnId = button == GLFW_MOUSE_BUTTON_LEFT ? 1 : button == GLFW_MOUSE_BUTTON_RIGHT ? 2 : 
          button == GLFW_MOUSE_BUTTON_MIDDLE ? 3 : 0;
  
        for (auto& c : render->mouseReleaseCallback)
          c(render->mouseX, render->mouseY, btnId);
      }
    });
    glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        auto render = static_cast<Render*>(glfwGetWindowUserPointer(window));
        for (auto& c : render->keyboardCallback)
          c(key, action);
      });

    for (auto& d : drawables)
    {
      d->Initialize();
    }

    GET_GL_ERROR("Render Initialize...\n");
}

Render::~Render()
{
  glfwTerminate();
}

void Render::StartLoop()
{
  while (!glfwWindowShouldClose(window)) 
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    OnDisplayStart();
    for (auto& d : drawables)
    {
      d->Draw();
    }
    OnDisplayEnd();

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
}