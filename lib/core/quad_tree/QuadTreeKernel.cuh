#pragma once

#include "lib/core/quad_tree/QuadTree.h"
#include "lib/tools/TreeTools.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// will divide at least 1 time
__global__ void QuadTreeKernel(float2* points, float2* pointsExch,
    QuadTree* tree, int depth, int maxDepth, int minPointsToDivide)
{
    // threads within a warp
    auto thisWarp = cg::coalesced_threads();
    auto thisBlock = cg::this_thread_block();
    
    const int warpsPerBlock = thisBlock.size() / warpSize;
    const int warpId = thisBlock.thread_rank() / warpSize;
    const int laneId = thisWarp.thread_rank() % warpSize;
    // to get a 'global' warp id
    // unsigned warpId2; 
    // asm volatile("mov.u32 %0, %warpid;" : "=r"(warpId2));

    QuadTree& subTree = tree[blockIdx.x];
    const auto aabb = subTree.bounds;
    const auto center = aabb.GetCenter();

    const int pointsCount = subTree.endId - subTree.startId;
    const int pointsPerWarp = (pointsCount + warpsPerBlock - 1) / warpsPerBlock;
    const int startId = subTree.startId + warpId * pointsPerWarp; 
    const int endId = min(startId + pointsPerWarp, subTree.endId);
    thisBlock.sync();

    if (pointsCount < minPointsToDivide || depth == maxDepth - 1)
    {
        // to be sure that the first array contains all the changes 
        // here pointsExch contains the latest chages, because
        // in the prev step Kernel got pointsExch as a first argument
        // **check a QuadTreeKernel call below
        if (depth > 0 && depth % 2 == 0) 
        {
            int start = subTree.startId;
            for (start += threadIdx.x; start < subTree.endId; start += thisBlock.size())
            {
                points[start] = pointsExch[start]; 
            }
        }

        return;
    }

    extern __shared__ int pointsInCell[];
    
    if (threadIdx.x < warpsPerBlock * 4)
        pointsInCell[threadIdx.x] = 0;

    int pointsInCellLocal[4] = {0, 0, 0, 0};

    thisBlock.sync();


    // each warp fills 4 subdivisions
    for (int i = startId + thisWarp.thread_rank(); 
             thisWarp.any(i < endId);
             i += thisWarp.size())
    {
        const auto isInRange = i < endId;

        const auto point = isInRange ? pointsExch[i] : float2{};

        const auto isUpLeft = isInRange && point.x <= center.x && point.y > center.y;
        // auto summ = __popc(thisWarp.ballot(isUpLeft));
        pointsInCellLocal[0] += __popc(thisWarp.ballot(isUpLeft));//thisWarp.shfl(summ, 0);

        const auto isUpRight = isInRange && point.x > center.x && point.y > center.y;
        // summ = __popc(thisWarp.ballot(isUpRight));
        pointsInCellLocal[1] += __popc(thisWarp.ballot(isUpRight));//thisWarp.shfl(summ, 0);

        const auto isDownLeft = isInRange && point.x <= center.x && point.y <= center.y;
        // summ = __popc(thisWarp.ballot(isDownLeft));
        pointsInCellLocal[2] += __popc(thisWarp.ballot(isDownLeft));//thisWarp.shfl(summ, 0);

        const auto isDownRight = isInRange && point.x > center.x && point.y <= center.y;
        // summ = __popc(thisWarp.ballot(isDownRight));
        pointsInCellLocal[3] += __popc(thisWarp.ballot(isDownRight));//thisWarp.shfl(summ, 0);

    }
    thisBlock.sync();

    // counts in each cell of each layer
    if (thisWarp.thread_rank() == 0)
    {
        pointsInCell[warpId * 4 + 0] = pointsInCellLocal[0];
        pointsInCell[warpId * 4 + 1] = pointsInCellLocal[1];
        pointsInCell[warpId * 4 + 2] = pointsInCellLocal[2];
        pointsInCell[warpId * 4 + 3] = pointsInCellLocal[3];
    }
    thisBlock.sync();

    // Store counts for each leaf in the latest index of a Shared Memory
    // one need atleat 4 warp to perform this action (fill each subdivision, i.e. leaf)
    if (warpId < 4)
    {
        // warpId - a cell number
        // thread_rank = a warp number (but max = 32!)
        int totalPointsCountPerCell = thisWarp.thread_rank() < warpsPerBlock
                        ? pointsInCell[thisWarp.thread_rank() * 4 + warpId]
                        : 0;

        // ccalc offset for each cell and for each warp
        for (int offset = 1; offset < warpsPerBlock; offset *= 2)
        {
            int n = thisWarp.shfl_up(totalPointsCountPerCell, offset);
            
            if (thisWarp.thread_rank() >= offset) 
                totalPointsCountPerCell += n;
        }

        thisWarp.sync();

        if (thisWarp.thread_rank() < warpsPerBlock)
        {
            pointsInCell[thisWarp.thread_rank() * 4 + warpId] = totalPointsCountPerCell;
        }
    }
    thisBlock.sync();

    // Calc endIds
    if (warpId == 0)
    {
        int itemsInCell = pointsInCell[(warpsPerBlock - 1) * 4 + 0];
        thisWarp.sync();

        for (int i = 1; i < 4; ++i)
        {
            int itemsCount = pointsInCell[(warpsPerBlock - 1) * 4 + i];
            thisWarp.sync();

            if (thisWarp.thread_rank() < warpsPerBlock)
            {
                pointsInCell[thisWarp.thread_rank() * 4 + i] += itemsInCell;
            }

            thisWarp.sync();

            itemsInCell += itemsCount;
        }
    }
    thisBlock.sync();

    int changeValue = 0;
    int changePointsId = 0;

    if (thisWarp.thread_rank() < warpsPerBlock)
    {
        // current global cell id
        changePointsId = thisWarp.thread_rank() * 4 + warpId;

        if (changePointsId != 0)
        {
            const int iddPrev = (changePointsId < 4) ? (warpsPerBlock - 1) * 4 + (changePointsId - 1) : (changePointsId - 4);

            changeValue = pointsInCell[iddPrev];
        }
        changeValue += subTree.startId;
    }

    thisBlock.sync();
    if (thisWarp.thread_rank() < warpsPerBlock)
    {
        pointsInCell[changePointsId] = changeValue;
    }
    thisBlock.sync();

    // 0-3 cell ids within a warp. i.e.: tl, tr, bl, br
    int offset1 = pointsInCell[warpId * 4 + 0];
    int offset2 = pointsInCell[warpId * 4 + 1];
    int offset3 = pointsInCell[warpId * 4 + 2];
    int offset4 = pointsInCell[warpId * 4 + 3];

    const int lane_mask_lt = (1 << laneId) - 1;
    thisBlock.sync();

    for (int i = startId + thisWarp.thread_rank(); 
             thisWarp.any(i < endId);
             i += thisWarp.size())
    {
        const auto isInRange = i < endId;
    
        const auto point = isInRange ? pointsExch[i] : float2{};

        const auto isUpLeft = isInRange && point.x <= center.x && point.y > center.y;
        const auto mask1 = thisWarp.ballot(isUpLeft);
        const auto destId1 = offset1 + __popc(mask1 & lane_mask_lt);
        if (isUpLeft)
        {
            points[destId1] = pointsExch[i];
        }
        offset1 += thisWarp.shfl(__popc(mask1), 0);

        const auto isUpRight = isInRange && point.x > center.x && point.y > center.y;
        const auto mask2 = thisWarp.ballot(isUpRight);
        const auto destId2 = offset2 + __popc(mask2 & lane_mask_lt);
        if (isUpRight)
        {
            points[destId2] = pointsExch[i];
        }
        offset2 += thisWarp.shfl(__popc(mask2), 0);

        const auto isDownLeft = isInRange && point.x <= center.x && point.y <= center.y;
        const auto mask3 = thisWarp.ballot(isDownLeft);
        const auto destId3 = offset3 + __popc(mask3 & lane_mask_lt);
        if (isDownLeft)
        {
            points[destId3] = pointsExch[i];
        }
        offset3 += thisWarp.shfl(__popc(mask3), 0);

        const auto isDownRight = isInRange && point.x > center.x && point.y <= center.y;
        const auto mask4 = thisWarp.ballot(isDownRight);
        const auto destId4 = offset4 + __popc(mask4 & lane_mask_lt);
        if (isDownRight)
        {
            points[destId4] = pointsExch[i];
        }
        offset4 += thisWarp.shfl(__popc(mask4), 0);
    }
    thisBlock.sync();

    if (thisBlock.thread_rank() == thisBlock.size() - 1)
    {
        const auto count = GetNodeByDepth<2>(depth) - (subTree.id & ~3);
        QuadTree* child = &tree[count];

        const auto treeIdNext = 4 * subTree.id;

        child[treeIdNext + 0].id = treeIdNext + 0;
        child[treeIdNext + 0].bounds.min = {subTree.bounds.min.x, center.y};
        child[treeIdNext + 0].bounds.max = {center.x, subTree.bounds.max.y};
        child[treeIdNext + 0].startId = subTree.startId;
        child[treeIdNext + 0].endId = offset1;

        child[treeIdNext + 1].id = treeIdNext + 1;
        child[treeIdNext + 1].bounds.min = center;
        child[treeIdNext + 1].bounds.max = subTree.bounds.max;
        child[treeIdNext + 1].startId = offset1;
        child[treeIdNext + 1].endId = offset2;

        child[treeIdNext + 2].id = treeIdNext + 2;
        child[treeIdNext + 2].bounds.min = subTree.bounds.min;
        child[treeIdNext + 2].bounds.max = center;
        child[treeIdNext + 2].startId = offset2;
        child[treeIdNext + 2].endId = offset3;

        child[treeIdNext + 3].id = treeIdNext + 3;
        child[treeIdNext + 3].bounds.min = {center.x, subTree.bounds.min.y};
        child[treeIdNext + 3].bounds.max = {subTree.bounds.max.x, center.y};
        child[treeIdNext + 3].startId = offset3;
        child[treeIdNext + 3].endId = offset4;

        QuadTreeKernel<<<4, thisBlock.size(), warpsPerBlock * 4 * sizeof(int)>>>(pointsExch, 
            points, &child[treeIdNext],
            depth + 1, maxDepth, minPointsToDivide);
    }
}