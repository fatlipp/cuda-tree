#pragma once

#include "lib/core/octree/Octree.h"
#include "lib/tools/TreeTools.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// will divide at least 1 time
__global__ void OctreeKernel(float3* points, float3* pointsExch,
    Octree* tree, int depth, int maxDepth, int minPointsToDivide)
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

    Octree& subTree = tree[blockIdx.x];
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
        // **check a OctreeKernel call below
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

    // size: warpsPerBlock * 8 
    extern __shared__ int pointsInCell[];
    
    if (threadIdx.x < warpsPerBlock * 8)
    {
        pointsInCell[threadIdx.x] = 0;
    }

    int pointsInCellLocal[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    thisBlock.sync();

    for (int i = startId + thisWarp.thread_rank(); 
             thisWarp.any(i < endId);
             i += thisWarp.size())
    {
        const auto isInRange = i < endId;
        const auto point = isInRange ? pointsExch[i] : float3{};
        const bool isFront = (point.z <= center.z);

        // front
        const auto isUpLeft = isInRange && isFront && point.x <= center.x && point.y > center.y;
        // auto summ = __popc(thisWarp.ballot(isUpLeft));
        pointsInCellLocal[0] += __popc(thisWarp.ballot(isUpLeft));//thisWarp.shfl(summ, 0);

        const auto isUpRight = isInRange && isFront && point.x > center.x && point.y > center.y;
        // summ = __popc(thisWarp.ballot(isUpRight));
        pointsInCellLocal[1] += __popc(thisWarp.ballot(isUpRight));//thisWarp.shfl(summ, 0);

        const auto isDownLeft = isInRange && isFront && point.x <= center.x && point.y <= center.y;
        // summ = __popc(thisWarp.ballot(isDownLeft));
        pointsInCellLocal[2] += __popc(thisWarp.ballot(isDownLeft));//thisWarp.shfl(summ, 0);

        const auto isDownRight = isInRange && isFront && point.x > center.x && point.y <= center.y;
        // summ = __popc(thisWarp.ballot(isDownRight));
        pointsInCellLocal[3] += __popc(thisWarp.ballot(isDownRight));//thisWarp.shfl(summ, 0);

        // back
        const auto isUpLeftBack = isInRange && !isFront && point.x <= center.x && point.y > center.y;
        // summ = __popc(thisWarp.ballot(isUpLeftBack));
        pointsInCellLocal[4] += __popc(thisWarp.ballot(isUpLeftBack));//thisWarp.shfl(summ, 0);

        const auto isUpRightBack = isInRange && !isFront && point.x > center.x && point.y > center.y;
        // summ = __popc(thisWarp.ballot(isUpRightBack));
        pointsInCellLocal[5] += __popc(thisWarp.ballot(isUpRightBack));//thisWarp.shfl(summ, 0);

        const auto isDownLeftBack = isInRange && !isFront && point.x <= center.x && point.y <= center.y;
        // summ = __popc(thisWarp.ballot(isDownLeftBack));
        pointsInCellLocal[6] += __popc(thisWarp.ballot(isDownLeftBack));//thisWarp.shfl(summ, 0);

        const auto isDownRightBack = isInRange && !isFront && point.x > center.x && point.y <= center.y;
        // summ = __popc(thisWarp.ballot(isDownRightBack));
        pointsInCellLocal[7] += __popc(thisWarp.ballot(isDownRightBack));//thisWarp.shfl(summ, 0);

    }
    thisBlock.sync();

    // counts in each cell of each layer
    if (thisWarp.thread_rank() == 0)
    {
        pointsInCell[warpId * 8 + 0] = pointsInCellLocal[0];
        pointsInCell[warpId * 8 + 1] = pointsInCellLocal[1];
        pointsInCell[warpId * 8 + 2] = pointsInCellLocal[2];
        pointsInCell[warpId * 8 + 3] = pointsInCellLocal[3];
        pointsInCell[warpId * 8 + 4] = pointsInCellLocal[4];
        pointsInCell[warpId * 8 + 5] = pointsInCellLocal[5];
        pointsInCell[warpId * 8 + 6] = pointsInCellLocal[6];
        pointsInCell[warpId * 8 + 7] = pointsInCellLocal[7];
    }
    thisBlock.sync();

    if (warpId < 8)
    {
        // warpId - a cell number
        // thread_rank = a warp number (but max = 32!)
        int totalPointsCountPerCell = thisWarp.thread_rank() < warpsPerBlock
                        ? pointsInCell[thisWarp.thread_rank() * 8 + warpId]
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
            pointsInCell[thisWarp.thread_rank() * 8 + warpId] = totalPointsCountPerCell;
        }
    }
    thisBlock.sync();
    // if (thisWarp.thread_rank() < warpsPerBlock)
    // {
    //     printf("depth: %d, warp: %d, TOTAL POINTS, val: %d\n", 
    //         thisWarp.thread_rank(), warpId, 
    //         pointsInCell[thisWarp.thread_rank() * 8 + warpId]
    //         );
    // }
    // thisBlock.sync();

    // ID = (warpId[0...N] * 8) + CELL_ID[tl, tr, bl, br]
    // last warpID - last CELL OFFSET for the ID
    // calc endIds
    if (warpId == 0)
    {
        int itemsInCell = pointsInCell[(warpsPerBlock - 1) * 8 + 0];
        thisWarp.sync();

        for (int i = 1; i < 8; ++i)
        {
            const int itemsCount = pointsInCell[(warpsPerBlock - 1) * 8 + i];
            thisWarp.sync();

            if (thisWarp.thread_rank() < warpsPerBlock)
            {
                pointsInCell[thisWarp.thread_rank() * 8 + i] += itemsInCell;
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
        changePointsId = thisWarp.thread_rank() * 8 + warpId;

        if (changePointsId != 0)
        {
            const int iddPrev = (changePointsId < 8) ? (warpsPerBlock - 1) * 8 + (changePointsId - 1) : (changePointsId - 8);

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

    int offsets[8] = {
            pointsInCell[warpId * 8 + 0],
            pointsInCell[warpId * 8 + 1],
            pointsInCell[warpId * 8 + 2],
            pointsInCell[warpId * 8 + 3],
            pointsInCell[warpId * 8 + 4],
            pointsInCell[warpId * 8 + 5],
            pointsInCell[warpId * 8 + 6],
            pointsInCell[warpId * 8 + 7],
        };
    // todo

    const int lane_mask_lt = (1 << laneId) - 1;
    thisBlock.sync();

    for (int i = startId + thisWarp.thread_rank(); 
             thisWarp.any(i < endId); // use any to prevent deadlock on a block sync
             i += thisWarp.size())
    {
        const auto isInRange = i < endId;
        const auto point = isInRange ? pointsExch[i] : float3{};
        const bool isFront = point.z <= center.z;

        // front
        {
            const auto isUpLeft = isInRange && isFront && point.x <= center.x && point.y > center.y;
            const auto mask1 = thisWarp.ballot(isUpLeft);
            const auto destId1 = offsets[0] + __popc(mask1 & lane_mask_lt);
            if (isUpLeft)
            {
                points[destId1] = pointsExch[i];
            }
            offsets[0] += thisWarp.shfl(__popc(mask1), 0);

            const auto isUpRight = isInRange && isFront && point.x > center.x && point.y > center.y;
            const auto mask2 = thisWarp.ballot(isUpRight);
            const auto destId2 = offsets[1] + __popc(mask2 & lane_mask_lt);
            if (isUpRight)
            {
                points[destId2] = pointsExch[i];
            }
            offsets[1] += thisWarp.shfl(__popc(mask2), 0);

            const auto isDownLeft = isInRange && isFront && point.x <= center.x && point.y <= center.y;
            const auto mask3 = thisWarp.ballot(isDownLeft);
            const auto destId3 = offsets[2] + __popc(mask3 & lane_mask_lt);
            if (isDownLeft)
            {
                points[destId3] = pointsExch[i];
            }
            offsets[2] += thisWarp.shfl(__popc(mask3), 0);

            const auto isDownRight = isInRange && isFront && point.x > center.x && point.y <= center.y;
            const auto mask4 = thisWarp.ballot(isDownRight);
            const auto destId4 = offsets[3] + __popc(mask4 & lane_mask_lt);
            if (isDownRight)
            {
                points[destId4] = pointsExch[i];
            }
            offsets[3] += thisWarp.shfl(__popc(mask4), 0);
        }

        // back
        {
            const auto isUpLeftBack = isInRange && !isFront && point.x <= center.x && point.y > center.y;
            const auto mask1 = thisWarp.ballot(isUpLeftBack);
            const auto destId1 = offsets[4] + __popc(mask1 & lane_mask_lt);
            if (isUpLeftBack)
            {
                points[destId1] = pointsExch[i];
            }
            offsets[4] += thisWarp.shfl(__popc(mask1), 0);

            const auto isUpRightBack = isInRange && !isFront && point.x > center.x && point.y > center.y;
            const auto mask2 = thisWarp.ballot(isUpRightBack);
            const auto destId2 = offsets[5] + __popc(mask2 & lane_mask_lt);
            if (isUpRightBack)
            {
                points[destId2] = pointsExch[i];
            }
            offsets[5] += thisWarp.shfl(__popc(mask2), 0);

            const auto isDownLeftBack = isInRange && !isFront && point.x <= center.x && point.y <= center.y;
            const auto mask3 = thisWarp.ballot(isDownLeftBack);
            const auto destId3 = offsets[6] + __popc(mask3 & lane_mask_lt);
            if (isDownLeftBack)
            {
                points[destId3] = pointsExch[i];
            }
            offsets[6] += thisWarp.shfl(__popc(mask3), 0);

            const auto isDownRightBack = isInRange && !isFront && point.x > center.x && point.y <= center.y;
            const auto mask4 = thisWarp.ballot(isDownRightBack);
            const auto destId4 = offsets[7] + __popc(mask4 & lane_mask_lt);
            if (isDownRightBack)
            {
                points[destId4] = pointsExch[i];
            }
            offsets[7] += thisWarp.shfl(__popc(mask4), 0);
        }
    }
    thisBlock.sync();

    if (thisBlock.thread_rank() == thisBlock.size() - 1)
    {
        const auto count = GetNodeByDepth<3>(depth) - (subTree.id & ~7);
        Octree* child = &tree[count];

        const auto treeIdNext = 8 * subTree.id;

        // front
        auto& child0 = child[treeIdNext + 0];
        child0.id = treeIdNext + 0;
        child0.bounds.min = {subTree.bounds.min.x, center.y, subTree.bounds.min.z};
        child0.bounds.max = {center.x, subTree.bounds.max.y, center.z};
        child0.startId = subTree.startId;
        child0.endId = offsets[0];

        auto& child1 = child[treeIdNext + 1];
        child1.id = treeIdNext + 1;
        child1.bounds.min = {center.x, center.y, subTree.bounds.min.z};
        child1.bounds.max = {subTree.bounds.max.x, subTree.bounds.max.y, center.z};
        child1.startId = offsets[0];
        child1.endId = offsets[1];

        auto& child2 = child[treeIdNext + 2];
        child2.id = treeIdNext + 2;
        child2.bounds.min = subTree.bounds.min;
        child2.bounds.max = center;
        child2.startId = offsets[1];
        child2.endId = offsets[2];

        auto& child3 = child[treeIdNext + 3];
        child3.id = treeIdNext + 3;
        child3.bounds.min = {center.x, subTree.bounds.min.y, subTree.bounds.min.z};
        child3.bounds.max = {subTree.bounds.max.x, center.y, center.z};
        child3.startId = offsets[2];
        child3.endId = offsets[3];

        /// back
        auto& child4 = child[treeIdNext + 4];
        child4.id = treeIdNext + 4;
        child4.bounds.min = {subTree.bounds.min.x, center.y, center.z};
        child4.bounds.max = {center.x, subTree.bounds.max.y, subTree.bounds.max.z};
        child4.startId = offsets[3];
        child4.endId = offsets[4];

        auto& child5 = child[treeIdNext + 5];
        child5.id = treeIdNext + 5;
        child5.bounds.min = center;
        child5.bounds.max = subTree.bounds.max;
        child5.startId = offsets[4];
        child5.endId = offsets[5];

        auto& child6 = child[treeIdNext + 6];
        child6.id = treeIdNext + 6;
        child6.bounds.min = {subTree.bounds.min.x, subTree.bounds.min.y, center.z};
        child6.bounds.max = {center.x, center.y, subTree.bounds.max.z};
        child6.startId = offsets[5];
        child6.endId = offsets[6];

        auto& child7 = child[treeIdNext + 7];
        child7.id = treeIdNext + 7;
        child7.bounds.min = {center.x, subTree.bounds.min.y, center.z};
        child7.bounds.max = {subTree.bounds.max.x, center.y, subTree.bounds.max.z};
        child7.startId = offsets[6];
        child7.endId = offsets[7];

        OctreeKernel<<<8, thisBlock.size(), warpsPerBlock * 8 * sizeof(int)>>>(pointsExch, 
            points, &child[treeIdNext],
            depth + 1, maxDepth, minPointsToDivide);
    }
}