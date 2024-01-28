#pragma once

#include "render/base/IDrawable.h"
#include "drawable/DrawableTreeTools.h"
#include "lib/config/TreeConfig.h"
#include "lib/core/ITree.h"

#include <glm/glm.hpp>
#include <GL/glew.h>

#include <vector>

template<typename T, int DIMENSIONS>
class DrawableTree : public IDrawable
{
public:
    DrawableTree(const ITree<T>* tree, const T* points, const TreeConfig& treeConfig, 
        const int pointSize, const int lineWidth);

public:
    void Initialize() override;
    void Draw() override;

private:
    const ITree<T>* tree;
    const T* points;
    const TreeConfig treeConfig;

    const int pointSize;
    const int lineWidth;

    std::vector<GLuint> treeVaoArr;
    std::vector<GLuint> treeVboArr;
    std::vector<GLuint> treeVcoArr;
    std::vector<GLuint> treeEboArr;

    std::vector<GLuint> pointVaoArr;
    std::vector<GLuint> pointVboArr;
    std::vector<GLuint> pointColors;
    std::vector<int> pointCounts;
};

template<typename T, int DIMENSIONS>
DrawableTree<T, DIMENSIONS>::DrawableTree(const ITree<T>* tree, const T* points, const TreeConfig& treeConfig, 
        const int pointSize, const int lineWidth)
        : tree { tree }
        , points { points }
        , treeConfig { treeConfig }
        , pointSize { pointSize }
        , lineWidth { lineWidth }
{
}

template<typename T, int DIMENSIONS>
void DrawableTree<T, DIMENSIONS>::Initialize()
{
    const auto [vao, vbo, vco, ebo] = CreateBO<T, DIMENSIONS>(tree, 0, 0);
    treeVaoArr.push_back(vao);
    treeVboArr.push_back(vbo);
    treeVcoArr.push_back(vco);
    treeEboArr.push_back(ebo);

    int id = 1;

    for (int depth = 0; depth < treeConfig.maxDepth; ++depth)
    {
        const auto leafs = GetNodeByDepth<DIMENSIONS>(depth);
        for (int leaf = 0; leaf < leafs; ++leaf)
        {
            const ITree<T>* const subTree = &tree[leaf];

            if ((subTree->PointsCount() < treeConfig.minPointsToDivide || 
                depth == treeConfig.maxDepth - 1) && subTree->PointsCount() > 0)
            {
                const auto [vao, vbo, vco, ebo] = CreateBO<T, DIMENSIONS>(subTree, depth, id);
                treeVaoArr.push_back(vao);
                treeVboArr.push_back(vbo);
                treeVcoArr.push_back(vco);
                treeEboArr.push_back(ebo);

                const auto [vaoPoint, vboPoint, vcoPoint, count] = CreatePointBO<T, DIMENSIONS>(subTree, points, depth, id);
                pointVaoArr.push_back(vaoPoint);
                pointVboArr.push_back(vboPoint);
                pointColors.push_back(vcoPoint);
                pointCounts.push_back(count);

                ++id;
            }
        }

        tree += leafs;
    }

    GET_GL_ERROR("DrawableTree() init");
}

template<typename T, int DIMENSIONS>
void DrawableTree<T, DIMENSIONS>::Draw()
{
    for (int i = 0; i < treeVaoArr.size(); ++i)
    {
        glBindVertexArray(treeVaoArr[i]);
            glLineWidth(lineWidth);
            glDrawElements(GL_LINE_STRIP, DIMENSIONS == 3 ? 16 : 5, GL_UNSIGNED_INT, 0);
            glLineWidth(1);
        glBindVertexArray(0);
    }
    GET_GL_ERROR("Draw() lines");

    for (int i = 0; i < pointVaoArr.size(); ++i)
    {
        glBindVertexArray(pointVaoArr[i]);
            glPointSize(pointSize);
            glDrawArrays(GL_POINTS, 0, pointCounts[i]);
            glPointSize(1);
        glBindVertexArray(0);
    }
    GET_GL_ERROR("Draw() points");
}