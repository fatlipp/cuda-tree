#pragma once

#include "lib/core/ITree.h"
#include "lib/core/CudaOperator.cuh"
#include "lib/tools/TreeTools.h"

#include <sstream>
#include <algorithm>

struct Neighbour
{
    int id;
    float distance;
};

template<typename T>
struct SerachCandidate
{
    const ITree<T>* tree;
    float dist;
    int depth;
};

template<typename T>
float CalcDistSqr(const T& p1, const T& p2)
{
    const T d1 = (p1 - p2);

    return norm(d1);
}

// == kNN ==
template<typename T>
void VisitCell(const ITree<T>* cell, TreeConfig config, const int depth, 
    const T& point, const int needCount,
    const T* points,
    std::vector<Neighbour>& result)
{
    if (cell == nullptr)
    {
        return;
    }

    if (cell->endId - cell->startId == 0)
    {
        return;
    }

    for (int i = cell->startId; i < cell->endId; ++i)
    {
        const float dist = CalcDistSqr(point, points[i]);

        if (result.size() < needCount)
        {
            result.push_back({i, dist});
            continue;
        }

        // replace the farthest item
        float maxLocalDist = dist;
        int maxLocalId = -1;
        for (int j = 0; j < result.size(); ++j)
        {
            if (maxLocalDist < result[j].distance)
            {
                maxLocalDist = result[j].distance;
                maxLocalId = j;
            }
        }

        if (maxLocalId > -1)
        {
            result[maxLocalId].id = i;
            result[maxLocalId].distance = dist;
        }
    }
}

template<typename T, int DIM>
void FillCandidates(const ITree<T>* localTree, TreeConfig config, const int depth,
    const T& point, const int needCount,
    const T* points,
    std::vector<SerachCandidate<T>>& candidates)
{
    if (depth == config.maxDepth - 1 && localTree->PointsCount() < config.minPointsToDivide)
    {
        candidates.push_back({localTree, CalcDistSqr(point, localTree->bounds.GetCenter()), depth});
        return;
    }

    for (int leaf = 0; leaf < 4; ++leaf)
    {
        const ITree<T>* subTree = &localTree[leaf];

        if (subTree->PointsCount() > 0)
        {
            const float dist = CalcDistSqr(point, subTree->bounds.GetCenter());

            if (depth < config.maxDepth - 1 && subTree->PointsCount() >= config.minPointsToDivide)
            {
                const auto nextId = GetNodeByDepth<DIM>(depth) + subTree->id * (DIM == 2 ? 4 : 8) - subTree->id;
                const ITree<T>* child = &subTree[nextId];
                
                FillCandidates<T, DIM>(child, config, depth + 1, point, needCount, points, candidates);
            }
            else
            {
                candidates.push_back({subTree, dist, depth});
            }
        }
    }
}

template<typename T, int DIM>
void CheckTree(const ITree<T>* localTree, const TreeConfig& config, const int depth,
    const T& point, const int needCount,
    const T* points,
    std::vector<Neighbour>& result)
{
    std::cout << "Fill candidates() ";

    auto satrtTime = std::chrono::high_resolution_clock::now();
    std::vector<SerachCandidate<T>> candidates;
    FillCandidates<T, DIM>(localTree, config, depth, point, needCount, points, candidates);
    auto endTime = std::chrono::high_resolution_clock::now();
    std::cout << "duration: " << 
        (std::chrono::duration_cast<std::chrono::milliseconds>(endTime - satrtTime).count()) << std::endl;

    std::cout << "Sort() ";
    satrtTime = std::chrono::high_resolution_clock::now();
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) 
        {
            return lhs.dist < rhs.dist;
        });
    endTime = std::chrono::high_resolution_clock::now();
    std::cout << "duration: " << 
        (std::chrono::duration_cast<std::chrono::milliseconds>(endTime - satrtTime).count()) << std::endl;


    std::cout << "Search() ";
    int idx = 0;
    satrtTime = std::chrono::high_resolution_clock::now();

    float distBuffer = -1.0f;

    // TODO: use sorted result to optimize calc dist
    while (idx < candidates.size())
    {
        VisitCell(candidates[idx].tree, config, candidates[idx].depth, point, needCount, points, result);

        float maxLocalDist = 0.0f;
        for (int j = 0; j < result.size(); ++j)
        {
            if (result[j].distance > maxLocalDist)
            {
                maxLocalDist = result[j].distance;
            }
        }

        if (maxLocalDist < candidates[idx].tree->bounds.Size().x) break;

        if (distBuffer >= 0.0f && (distBuffer <= maxLocalDist * 0.7f)) break;

        // to avoid overchecking (bug: if cell is big enough, on corners will not find NN)
        // try to use dist
        if (idx % 100000 == 0)
        {
            distBuffer = maxLocalDist;
        }

        ++idx;
    }
    endTime = std::chrono::high_resolution_clock::now();
    std::cout << "duration: " << 
        (std::chrono::duration_cast<std::chrono::milliseconds>(endTime - satrtTime).count()) << std::endl;
}

template<typename T, int DIM>
std::vector<Neighbour> NearestNeighbours(const ITree<T>* tree, const TreeConfig& config,
    const T& point, const int count, const T* points)
{
    std::cout << "kNN(p:{" << point.x << ", " << point.y << "}, count: " << count << ")" << std::endl;
    if (!tree->Check(point))
    {
        std::cout << "A point is out of BBox!\n\n";
        return {};
    }

    std::vector<Neighbour> result;

    if (config.maxDepth <= 1 || (tree->endId - tree->startId) < config.minPointsToDivide)
    {
        CheckTree<T, DIM>(tree, config, 0, point, count, points, result);

        return result;
    }

    auto satrtTime = std::chrono::high_resolution_clock::now();
    CheckTree<T, DIM>(tree + 1, config, 1, point, count, points, result);
    auto endTime = std::chrono::high_resolution_clock::now();
    std::cout << "KNN() duration: " << 
        (std::chrono::duration_cast<std::chrono::milliseconds>(endTime - satrtTime).count()) << std::endl;

    return result;
}

// == radius search ==
template<typename T>
bool CheckRadius(const ITree<T>* cell, const T& point, const float r)
{
    return false;
}

template<>
bool CheckRadius<float2>(const ITree<float2>* tree, const float2& point, const float r)
{
    const auto& center = tree->bounds.GetCenter();
    const auto circleDistanceX = abs(point.x - center.x);
    const auto circleDistanceY = abs(point.y - center.y);

    const auto size = tree->bounds.Size() / 2.0f;

    if (circleDistanceX > (size.x + r)) { return false; }
    if (circleDistanceY > (size.y + r)) { return false; }

    if (circleDistanceX <= (size.x)) { return true; } 
    if (circleDistanceY <= (size.y)) { return true; }

    const auto dx = (circleDistanceX - size.x);
    const auto dy = (circleDistanceY - size.y);
    const auto distanceSqr = dx * dx + dy * dy;

    return distanceSqr <= (r * r);
}

template<>
bool CheckRadius<float3>(const ITree<float3>* tree, const float3& point, const float r)
{
    const auto& center = tree->bounds.GetCenter();
    const auto circleDistanceX = abs(point.x - center.x);
    const auto circleDistanceY = abs(point.y - center.y);
    const auto circleDistanceZ = abs(point.z - center.z);

    const auto size = tree->bounds.Size() / 2.0f;

    if (circleDistanceX > (size.x + r)) { return false; }
    if (circleDistanceY > (size.y + r)) { return false; }
    if (circleDistanceZ > (size.z + r)) { return false; }

    if (circleDistanceX <= (size.x)) { return true; } 
    if (circleDistanceY <= (size.y)) { return true; }
    if (circleDistanceZ <= (size.z)) { return true; }

    const auto dx = (circleDistanceX - size.x);
    const auto dy = (circleDistanceY - size.y);
    const auto dz = (circleDistanceZ - size.z);
    const auto distanceSqr = dx * dx + dy * dy + dz * dz;

    return distanceSqr <= (r * r);
}

template<typename T>
void VisitCellRadius(const ITree<T>* cell, TreeConfig config, const int depth, 
    const T& center, const float radius,
    const T* points,
    std::vector<Neighbour>& result)
{
    if (cell == nullptr || cell->PointsCount() == 0)
    {
        return;
    }

    const float radiusSqr = radius * radius;

    for (int i = cell->startId; i < cell->endId; ++i)
    {
        const float dist = CalcDistSqr(center, points[i]);

        if (dist < radiusSqr)
        {
            result.push_back({i, 0});
        }
    }
}

template<typename T, int DIM>
void FillCandidatesRadius(const ITree<T>* localTree, const TreeConfig& config, const int depth,
    const T& center, const float radius,
    const T* points, std::vector<SerachCandidate<T>>& candidates)
{
    for (int leaf = 0; leaf < 4; ++leaf)
    {
        const ITree<T>* subTree = &localTree[leaf];

        if (subTree->PointsCount() > 0 && CheckRadius(subTree, center, radius))
        {
            if (depth < config.maxDepth - 1 && subTree->PointsCount() >= config.minPointsToDivide)
            {
                const auto nextId = GetNodeByDepth<DIM>(depth) + subTree->id * (DIM == 2 ? 4 : 8) - subTree->id;
                
                FillCandidatesRadius<T, DIM>(&subTree[nextId], config, depth + 1, center, radius, points, candidates);
            }
            else
            {
                candidates.push_back({subTree, -1, depth});
            }
        }
    }
}

template<typename T, int DIM>
void CheckTreeRadius(const ITree<T>* localTree, const TreeConfig& config, const int depth,
    const T& center, const float radius,
    const T* points,
    std::vector<Neighbour>& result)
{
    std::cout << "Fill candidates() ";

    auto satrtTime = std::chrono::high_resolution_clock::now();
    std::vector<SerachCandidate<T>> candidates;
    FillCandidatesRadius<T, DIM>(localTree, config, depth, center, radius, points, candidates);
    auto endTime = std::chrono::high_resolution_clock::now();
    std::cout << "candidates = " << candidates.size() << std::endl;
    std::cout << "duration: " << 
        (std::chrono::duration_cast<std::chrono::milliseconds>(endTime - satrtTime).count()) << std::endl;

    std::cout << "Search() ";
    satrtTime = std::chrono::high_resolution_clock::now();

    int idx = 0;
    while (idx < candidates.size())
    {
        VisitCellRadius(candidates[idx].tree, config, candidates[idx].depth, center, radius, points, result);

        ++idx;
    }
    endTime = std::chrono::high_resolution_clock::now();
    std::cout << "duration: " << 
        (std::chrono::duration_cast<std::chrono::milliseconds>(endTime - satrtTime).count()) << std::endl;
}

template<typename T, int DIM>
std::vector<Neighbour> RadiusSearch(const ITree<T>* tree, const TreeConfig& config, 
    const T& center, const float radius, const T* points)
{
    std::cout << "RadiusSearch(p:{" << center.x << ", " << center.y << "}, r: " << radius << ")" << std::endl;
    
    const auto satrtTime = std::chrono::high_resolution_clock::now();

    if (!CheckRadius(tree, center, radius))
    {
        std::cout << "A points is outside of the tree\n";
        return {};
    }

    std::vector<Neighbour> result;

    if (config.maxDepth <= 1 || tree->PointsCount() < config.minPointsToDivide)
    {
        VisitCellRadius(tree, config, 0, center, radius, points, result);
    }
    else
    {
        CheckTreeRadius<T, DIM>(tree + 1, config, 1, center, radius, points, result);
    }

    const auto endTime = std::chrono::high_resolution_clock::now();
    std::cout << "RadiusSearch() duration: " << 
        (std::chrono::duration_cast<std::chrono::milliseconds>(endTime - satrtTime).count()) << std::endl;

    return result;
}
