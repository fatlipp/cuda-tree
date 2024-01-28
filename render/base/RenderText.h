#pragma once

#include <string>
#include <vector>

#include <GL/gl.h>
#include <GL/glut.h>

class RenderText
{
struct TextData
{
   int x;
   int y;
   std::string text;
   Vec3 color;
};

public:
   void AddText(const int x, const int y, const std::string& text, 
      const Vec3& color = {1.0f, 1.0f, 1.0f})
   {
      texts.push_back({
            .x = x,
            .y = y,
            .text = text,
            .color = color
         });
   }

   void Clear()
   {
      texts.clear();
   }

   void Draw(const float width, const float height)
   {
      glMatrixMode(GL_MODELVIEW);
      glPushMatrix();
      glLoadIdentity();    

      glMatrixMode( GL_PROJECTION );
      glPushMatrix();
      glLoadIdentity();
      gluOrtho2D(0, width, 0, height);

      for (const auto& [x, y, text, color] : texts)
      {
         glColor3f(color.x, color.y, color.z);
         DrawText2(x, height - y, text.c_str());
      }

      glPopMatrix();

      glMatrixMode(GL_MODELVIEW);
      glPopMatrix();
   }

private:
   void DrawText2(float x, float y, const char* text) 
   {
      glRasterPos2i(x, y);
      for (const char* c = text; *c != '\0'; c++)
      {
         glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c);
      }
   }

private:
   std::vector<TextData> texts;
};