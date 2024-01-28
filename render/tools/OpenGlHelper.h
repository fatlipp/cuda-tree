#pragma once

void getLastGlError(const char *errorMessage, const char *file, const int line);

#define GET_GL_ERROR(msg) getLastGlError(msg, __FILE__, __LINE__);