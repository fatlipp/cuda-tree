#include "drawable/DrawableTree.h"
#include "render/tools/OpenGlHelper.h"
#include "lib/tools/TreeTools.h"
#include "lib/core/ITree.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <tuple>

template<typename T, int DIMENSIONS>
std::tuple<GLuint, GLuint, GLuint, GLuint> CreateBO(const ITree<T>* tree, const int depth, const int id)
{
    return {0, 0, 0, 0};
}

template<typename T, int DIMENSIONS>
std::tuple<GLuint, GLuint, int, GLuint> CreatePointBO(const ITree<T>* tree, const T* pointsIn, const int depth, const int id)
{
    return {0, 0, 0, 0};
}

static std::vector<float3> colors;

float3 GetColor(const int id)
{
    if (colors.size() < id + 1)
    {
        colors.push_back({
            (static_cast<float>(rand()) / RAND_MAX) * 1.0f + 0.1f,
            (static_cast<float>(rand()) / RAND_MAX) * 1.0f + 0.1f,
            (static_cast<float>(rand()) / RAND_MAX) * 1.0f + 0.1f
        });
    }

    return colors[id];
}

template<>
std::tuple<GLuint, GLuint, GLuint, GLuint> CreateBO<float3, 3>(const ITree<float3>* tree, const int depth, const int id)
{
    GLuint treeVbo;
    GLuint treeVao;
    GLuint treeVco;
    GLuint treeEbo;

    glGenVertexArrays(1, &treeVao);
    glGenBuffers(1, &treeVbo);
    glGenBuffers(1, &treeVco);
    glGenBuffers(1, &treeEbo);

    const auto& bounds = tree->bounds;

    float3 cube[8];
    cube[0] = {bounds.min.x, bounds.min.y, bounds.min.z};
    cube[1] = {bounds.min.x, bounds.max.y, bounds.min.z};
    cube[2] = {bounds.max.x, bounds.max.y, bounds.min.z};
    cube[3] = {bounds.max.x, bounds.min.y, bounds.min.z};
    cube[4] = {bounds.min.x, bounds.min.y, bounds.max.z};
    cube[5] = {bounds.min.x, bounds.max.y, bounds.max.z};
    cube[6] = {bounds.max.x, bounds.max.y, bounds.max.z};
    cube[7] = {bounds.max.x, bounds.min.y, bounds.max.z};
    
    const auto col = GetColor(id);

    float3 color[8];
    // std::memset(color, col, float3);
    color[0] = col;
    color[1] = col;
    color[2] = col;
    color[3] = col;
    color[4] = col;
    color[5] = col;
    color[6] = col;
    color[7] = col;

    const unsigned indices[] = {
        0, 1, 2, 3, 0,
        4, 5, 6, 7, 4,
        5, 1, 2, 6, 7, 
        3
    };

    glBindVertexArray(treeVao);
        // ids (16)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, treeEbo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        // pos (8)
        glBindBuffer(GL_ARRAY_BUFFER, treeVbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * 8, cube, GL_STATIC_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3), 0);
            glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        // color (8)
        glBindBuffer(GL_ARRAY_BUFFER, treeVco);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * 8, color, GL_STATIC_DRAW);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float3), 0);
            glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return {treeVao, treeVbo, treeVco, treeEbo};
}

template<>
std::tuple<GLuint, GLuint, int, GLuint> CreatePointBO<float3, 3>(const ITree<float3>* tree, const float3* pointsIn, 
    const int depth, const int id)
{
    float3 points[tree->PointsCount()];
    float3 pointColors[tree->PointsCount()];

    for (int i = tree->startId; i < tree->endId; ++i)
    {
        points[i - tree->startId] = pointsIn[i];
        pointColors[i - tree->startId] = GetColor(id);
    }

    GLuint vao;
    GLuint vbo;
    GLuint vco;

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &vco);

    glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * tree->PointsCount(), points, GL_STATIC_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3), 0);
            glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, vco);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * tree->PointsCount(), pointColors, GL_STATIC_DRAW);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float3), 0);
            glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return {vao, vbo, vco, tree->PointsCount()};
}

template<>
std::tuple<GLuint, GLuint, GLuint, GLuint> CreateBO<float2, 2>(const ITree<float2>* tree, const int depth, const int id)
{
    GLuint treeVbo;
    GLuint treeVao;
    GLuint treeVco;
    GLuint treeEbo;

    glGenVertexArrays(1, &treeVao);
    glGenBuffers(1, &treeVbo);
    glGenBuffers(1, &treeVco);
    glGenBuffers(1, &treeEbo);

    float2 cube[4];
    cube[0] = {tree->bounds.min.x, tree->bounds.min.y};
    cube[1] = {tree->bounds.min.x, tree->bounds.max.y};
    cube[2] = {tree->bounds.max.x, tree->bounds.max.y};
    cube[3] = {tree->bounds.max.x, tree->bounds.min.y};
    
    const auto col = GetColor(id);
    float3 color[4];
    color[0] = col;
    color[1] = col;
    color[2] = col;
    color[3] = col;

    const unsigned indices[] = {
        0, 1, 2, 3, 0
    };

    glBindVertexArray(treeVao);
        // ids (16)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, treeEbo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        // pos (4)
        glBindBuffer(GL_ARRAY_BUFFER, treeVbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * 4, cube, GL_STATIC_DRAW);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float2), 0);
            glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        // color (4)
        glBindBuffer(GL_ARRAY_BUFFER, treeVco);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * 4, color, GL_STATIC_DRAW);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float3), 0);
            glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return {treeVao, treeVbo, treeVco, treeEbo};
}

template<>
std::tuple<GLuint, GLuint, int, GLuint> CreatePointBO<float2, 2>(const ITree<float2>* tree, const float2* pointsIn, const int depth, const int id)
{
    float2 points[tree->PointsCount()];
    float3 pointColors[tree->PointsCount()];

    for (int i = tree->startId; i < tree->endId; ++i)
    {
        points[i - tree->startId] = pointsIn[i];
        pointColors[i - tree->startId] = GetColor(id);
    }

    GLuint vao;
    GLuint vbo;
    GLuint vco;

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &vco);

    glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * tree->PointsCount(), points, GL_STATIC_DRAW);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float2), 0);
            glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, vco);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * tree->PointsCount(), pointColors, GL_STATIC_DRAW);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float3), 0);
            glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return {vao, vbo, vco, tree->PointsCount()};
}