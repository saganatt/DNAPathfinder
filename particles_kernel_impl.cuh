/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * CUDA particle system kernel code.
 */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <math.h>
//#include <float.h> // for FLT_MAX if needed
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"
#include "cluster.h"

#include "Bresenham.h"

#if USE_TEX
// texture for particle position
texture<float3, 1, cudaReadModeElementType> oldPosTex;

texture<uint, 1, cudaReadModeElementType> gridParticleHashTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;
#endif

// simulation parameters in constant memory
__constant__ SimParams params;

// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
    gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
    gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (params.gridSize.y-1);
    gridPos.z = gridPos.z & (params.gridSize.z-1);
    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__
void calcHashD(uint   *gridParticleHash,  // output
               uint   *gridParticleIndex, // output
               float3 *pos,               // input: positions
               uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    volatile float3 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(p);
    uint hash = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint   *cellStart,        // output: cell start index
                                  uint   *cellEnd,          // output: cell end index
                                  float3 *sortedPos,        // output: sorted positions
                                  uint   *gridParticleHash, // input: sorted grid hashes
                                  uint   *gridParticleIndex,// input: sorted particle indices
                                  float3 *oldPos,           // input: sorted position array
                                  uint    numParticles)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

    uint hash;

    // handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x+1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index-1];
        }
    }

    cg::sync(cta);

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

        // Now use the sorted index to reorder the pos data
        uint sortedIndex = gridParticleIndex[index];
        float3 pos = FETCH(oldPos, sortedIndex);

        sortedPos[index] = pos;
    }
}

// Assumes that row and column start from 0
__device__
int32_t getAdjTriangleEntry(int32_t *adjTriangle,
                            uint numParticles,
                            uint32_t row,
                            uint32_t column) {
    if(row > column) {
        uint32_t tmp = row;
        row = column;
        column = tmp;
    }
    return adjTriangle[(row * (2 * numParticles - 1 - row)) / 2 + column - row - 1];
};

// Assumes that row and column start from 0
__device__
void setAdjTriangleEntry(int32_t *adjTriangle,
                         uint numParticles,
                         uint32_t row,
                         uint32_t column,
                         int32_t value) {
    if(row > column) {
        uint32_t tmp = row;
        row = column;
        column = tmp;
    }
    adjTriangle[(row * (2 * numParticles - 1 - row)) / 2 + column - row - 1] = value;
};

// Assumes that row and column start from 0
__device__
void addAdjTriangleEntry(int32_t *adjTriangle,
                         uint numParticles,
                         uint32_t row,
                         uint32_t column) {
    if(row > column) {
        uint32_t tmp = row;
        row = column;
        column = tmp;
    }
    atomicAdd(&adjTriangle[(row * (2 * numParticles - 1 - row)) / 2 + column - row - 1], 1);
};

// Assumes that row and column start from 0
__device__
void getPairFromAdjTriangleIndex(uint numParticles,
                                 uint32_t index,
                                 uint32_t *row,
                                 uint32_t *column) {
    float b = 1.0f - 2.0f * numParticles;
    float c = 8.0f * index;
    float delta = powf(b, 2.0f) - c;
    float deltasq = powf(delta, 0.5f);
    float fraction = (-b - deltasq) / 2.0f;
    *row = (uint32_t)floorf(fraction);
    *column = index + *row + 1 - (*row * (2 * numParticles - *row - 1)) / 2;
}

// test all possible pairs against contour borders
__global__
void checkContourD(int32_t *adjTriangle,      // output: adjacency triangle
                   uint32_t adjTriangleSize,
                   float3 *oldPos,           // input: positions
                   uint32_t *contour,        // input: contour
                   uint3 contourSize,
                   uint   numParticles) {
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= adjTriangleSize) return;

    uint32_t ind1, ind2;
    getPairFromAdjTriangleIndex(numParticles, index, &ind1, &ind2);

    // read particle data from sorted arrays
    float3 pos1 = FETCH(oldPos, ind1);
    float3 pos2 = FETCH(oldPos, ind2);

    if(!checkPathInContour(pos1, pos2, contour, contourSize, params.voxelSize)) {
        setAdjTriangleEntry(adjTriangle, numParticles, ind1, ind2, -1);
    }
}

// compare distances to all particles in a given cell
__device__
void compareClosest(int3    gridPos,
                    uint    index,
                    float3  pos,
                    uint32_t *first_ind,
                    uint32_t *second_ind,
                    float *first_dist_sq,
                    float *second_dist_sq,
                    float searchRadius,
                    float3 *sortedPos,
                    int32_t *adjTriangle,
                    uint numParticles,
                    uint   *cellStart,
                    uint   *cellEnd)
{
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = FETCH(cellStart, gridHash);

    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = FETCH(cellEnd, gridHash);

        for (uint j=startIndex; j<endIndex; j++)
        {
            if (j != index)                 // check not comparing with self
            {
                float3 pos2 = FETCH(sortedPos, j);

                float dist_sq = dot(pos - pos2, pos - pos2);

                if(getAdjTriangleEntry(adjTriangle, numParticles, index, j) != -1) {
                    if (dist_sq < *first_dist_sq) {
                        *first_dist_sq = dist_sq;
                        *first_ind = j;
                    }
                    else if (j != *first_ind && dist_sq < *second_dist_sq) {
                        *second_dist_sq = dist_sq;
                        *second_ind = j;
                    }
                }
            }
        }
    }
}

__global__
void connectPairsD(int32_t *adjTriangle,      // output: adjacency triangle
                  float3 *sortedPos,          // input: sorted positions
                  uint *gridParticleIndex,   // input: sorted particle indices
                  uint *cellStart,
                  uint *cellEnd,
                  uint numParticles)
{
    uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    // read particle data from sorted arrays
    uint sortedIndex = gridParticleIndex[index]; // original particle index
    float3 pos = FETCH(sortedPos, index);

    // get address in grid
    int3 gridPos = calcGridPos(pos);

    float searchRadiusInParticleRadius = params.searchRadius / params.particleRadius;
    int searchRadiusInCellsNumber = (int) ceilf((searchRadiusInParticleRadius - 1.0f) / 2.0f);

    // examine neighbouring cells
    uint32_t first_ind = numParticles; // particles are indexed from 0
    uint32_t second_ind = numParticles;
    float first_dist_sq = params.searchRadius * params.searchRadius;
    float second_dist_sq = params.searchRadius * params.searchRadius;

    for (int z = -searchRadiusInCellsNumber; z <= searchRadiusInCellsNumber; z++) {
        for (int y = -searchRadiusInCellsNumber; y <= searchRadiusInCellsNumber; y++) {
            for (int x = -searchRadiusInCellsNumber; x <= searchRadiusInCellsNumber; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                compareClosest(neighbourPos, index, pos, &first_ind, &second_ind,
                               &first_dist_sq, &second_dist_sq, params.searchRadius, sortedPos,
                               adjTriangle, numParticles, cellStart, cellEnd);
            }
        }
    }

    // write new pairs
    if (first_ind < numParticles) {
        uint sorted_first_ind = gridParticleIndex[first_ind]; // original particle index
        setAdjTriangleEntry(adjTriangle, numParticles, sortedIndex, sorted_first_ind, 1);
    }
    if (second_ind < numParticles) {
        uint sorted_second_ind = gridParticleIndex[second_ind]; // original particle index
        setAdjTriangleEntry(adjTriangle, numParticles, sortedIndex, sorted_second_ind, 1);
    }
}

__global__
void calcDegreesD(int32_t *adjTriangle,       // output: adjacency triangle
                 int32_t *edgesCount,        // output: number of edges
                 int32_t *degrees,           // output: vertices degrees
                 uint   numParticles)
{
    uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;

    degrees[index] = 0;
    for(int i = 0; i < index; i++) {
        if(getAdjTriangleEntry(adjTriangle, numParticles, i, index) > 0) {
            atomicAdd(edgesCount, 1);
            degrees[index] += 1;
        }
    }
    for(int i = index + 1; i < numParticles; i++) {
        if(getAdjTriangleEntry(adjTriangle, numParticles, index, i) > 0) {
            atomicAdd(edgesCount, 1);
            degrees[index] += 1;
        }
    }
}

__global__
void markIsolatedVerticesD(int32_t *degrees,           // output: vertices degrees
                          bool *isolatedVertices,     // output: isolated vertices
                          uint   numParticles)
{
    uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= numParticles) return;

    if(degrees[index] == 0) {
        isolatedVertices[index] = true;
    }
}

__global__
void createAdjListD(int32_t *adjacencyList,      // output: adjacency list
                    int32_t *adjTriangle,        // input: adjacency triangle
                    int32_t *edgesOffset,        // input: scanned degrees
                    int32_t *edgesSize,          // input: vertices degrees
                    uint numParticles)           // input: number of vertices)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles) return;

    int currentPos = edgesOffset[index];
    for(int i = 0; i < index; i++)
    {
        if(getAdjTriangleEntry(adjTriangle, numParticles, index, i) > 0)
        {
            if(currentPos >= edgesOffset[index] + edgesSize[index])
            {
                printf("[%d] Create adj list: index out of bonds: %d %d %d!\n", index, currentPos, edgesOffset[index], edgesSize[index]);
            }
            adjacencyList[currentPos] = i;
            currentPos++;
        }
    }
    for(int i = index + 1; i < numParticles; i++)
    {
        if(getAdjTriangleEntry(adjTriangle, numParticles, index, i) > 0)
        {
            if(currentPos >= edgesOffset[index] + edgesSize[index])
            {
                printf("[%d] Create adj list: index out of bonds: %d %d %d!\n", index, currentPos, edgesOffset[index], edgesSize[index]);
            }
            adjacencyList[currentPos] = i;
            currentPos++;
        }
    }
}

__global__
void readAdjListD(int32_t *adjacencyList,      // output: adjacency list
                  int32_t *edgesOffset,        // input: scanned degrees
                  int32_t *edgesSize,          // input: vertices degrees
                  uint numParticles)           // input: number of vertices)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles) return;

    for(int i = edgesOffset[index]; i < edgesOffset[index] + edgesSize[index]; i++) {
        printf("[%d] %d\n", index, adjacencyList[i]);
    }
}

__global__
void completeClusterStatsD(int32_t *edgesSize,
                           float3 *oldPos,
                           int numParticles,
                           bool *frontier,
                           Cluster *cluster)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles) return;

    if(frontier[index]) {
        atomicAdd(&(cluster->clusterSize), 1);
        if(edgesSize[index] == 1) {
            atomicAdd(&(cluster->leavesCount), 1);
        }
        else if(edgesSize[index] > 2) {
            atomicAdd(&(cluster->branchingsCount), 1);
        }
        atomicAdd(&(cluster->massCentre.x), oldPos[index].x);
        atomicAdd(&(cluster->massCentre.x), oldPos[index].y);
        atomicAdd(&(cluster->massCentre.x), oldPos[index].z);
    }
}

#endif