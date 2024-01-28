#include <iostream>

#include <GL/glew.h>

void getLastGlError(const char *errorMessage, const char *file, const int line) 
{                               
  GLenum gl_error = glGetError();

  if (gl_error != GL_NO_ERROR) 
  {
      fprintf(stderr, "GL Error in file '%s' in line %d :\n", file, line);
      fprintf(stderr, "%s (%s)", gluErrorString(gl_error), errorMessage);
      fprintf(stderr, ", id: %d\n", gl_error);
      exit(EXIT_FAILURE);
  }
}