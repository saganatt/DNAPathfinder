#ifndef __PARTICLES_KERNELS_H__
#define __PARTICLES_KERNELS_H__

#include <stdio.h>
#include <math.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include <helper_math.h>
#include <math_constants.h>

#include "kernelParams.cuh"
#include "Bresenham.cuh"

// Kernel constants in constant memory
__constant__ KernelParams params;

// Functions from NVIDIA's particles sample

// Calculate position in uniform grid
__device__ int3 calcGridPos(float3 p) {
    int3 gridPos;
    gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
    gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
    gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
    return gridPos;
}

// Calculate address in grid from position (clamping to edges)
__device__ uint32_t calcGridHash(int3 gridPos) {
    gridPos.x = gridPos.x & (params.gridSize.x - 1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (params.gridSize.y - 1);
    gridPos.z = gridPos.z & (params.gridSize.z - 1);
    return gridPos.z * params.gridSize.y * params.gridSize.x + gridPos.y * params.gridSize.x + gridPos.x;
}

// Calculate grid hash value for each particle
__global__
void calcHashD(uint32_t *gridParticleHash,                      // output: sorted grid hashes
               uint32_t *gridParticleIndex,                     // output: sorted particle indices
               float3 *pos) {                                   // input: positions
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= params.numParticles) return;

    volatile float3 p = pos[index];

    // Get address in grid
    int3 gridPos = calcGridPos(p);
    uint32_t hash = calcGridHash(gridPos);

    // Store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

// Rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint32_t *cellStart,          // output: cell start index
                                  uint32_t *cellEnd,            // output: cell end index
                                  float3 *sortedPos,            // output: sorted particle positions
                                  uint32_t *gridParticleHash,   // input: sorted grid hashes
                                  uint32_t *gridParticleIndex,  // input: sorted particle indices
                                  float3 *oldPos) {             // input: particle positions
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ uint32_t sharedHash[];    // blockSize + 1 elements
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t hash;

    // Handle case when no. of particles not multiple of block size
    if (index < params.numParticles) {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x + 1] = hash;

        if (index > 0 && threadIdx.x == 0) {
            // First thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index - 1];
        }
    }

    cg::sync(cta);

    if (index < params.numParticles) {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell
        if (index == 0 || hash != sharedHash[threadIdx.x]) {
            cellStart[hash] = index;

            if (index > 0) {
                cellEnd[sharedHash[threadIdx.x]] = index;
            }
        }

        if (index == params.numParticles - 1) {
            cellEnd[hash] = index + 1;
        }

        // Now use the sorted index to reorder the pos data
        uint32_t sortedIndex = gridParticleIndex[index];
        float3 pos = oldPos[sortedIndex];

        sortedPos[index] = pos;
    }
}

// Custom functions

// Returns adjacency triangle entry for given pairs of vertices
// Assumes that row and column start from 0
__device__
int32_t getAdjTriangleEntry(int32_t *adjTriangle,       // input: adjacency triangle
                            uint32_t row,               // input: first vertex
                            uint32_t column) {          // input: second vertex
    if (row > column) {
        uint32_t tmp = row;
        row = column;
        column = tmp;
    }
    return adjTriangle[(row * (2 * params.numParticles - 1 - row)) / 2 + column - row - 1];
};

// Sets adjacency triangle entry for given pairs of vertices
// Assumes that row and column start from 0
__device__
void setAdjTriangleEntry(int32_t *adjTriangle,          // input: adjacency triangle
                         uint32_t row,                  // input: first vertex
                         uint32_t column,               // input: second vertex
                         int32_t value) {               // input: value to be set
    if (row > column) {
        uint32_t tmp = row;
        row = column;
        column = tmp;
    }
    adjTriangle[(row * (2 * params.numParticles - 1 - row)) / 2 + column - row - 1] = value;
};

// Converts single index of adjacency triangle to the corresponding pair of vertices
// Assumes that row and column start from 0
__device__
void getPairFromAdjTriangleIndex(uint32_t *row,         // output: first vertex
                                 uint32_t *column,      // output: second vertex
                                 uint32_t index) {      // input: index in adjacency triangle
    float b = 1.0f - 2.0f * params.numParticles;
    float c = 8.0f * index;
    float delta = powf(b, 2.0f) - c;
    float deltasq = powf(delta, 0.5f);
    float fraction = (-b - deltasq) / 2.0f;
    *row = (uint32_t)floorf(fraction);
    *column = index + *row + 1 - (*row * (2 * params.numParticles - *row - 1)) / 2;
}

// Test all possible pairs against contour borders
__global__
void checkContourD(int32_t *adjTriangle,                // output: adjacency triangle
                   uint32_t adjTriangleSize,            // input: adjacency triangle size
                   float3 *oldPos,                      // input: particles positions
                   uint32_t *contour,                   // input: contour array
                   uint3 contourSize) {                 // input: contour size
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= adjTriangleSize) return;

    uint32_t ind1, ind2;
    getPairFromAdjTriangleIndex(&ind1, &ind2, index);
    float3 pos1 = oldPos[ind1];
    float3 pos2 = oldPos[ind2];

    if (!checkPathInContour(pos1, pos2, contour, contourSize, params.voxelSize)) {
        setAdjTriangleEntry(adjTriangle, ind1, ind2, -1);
    }
}

// Functors with different conditions for the elements that will be connected
// (particles / clusters)
struct checkParticleParticleFunctor {
    __device__
    bool operator()(uint32_t v1, uint32_t v2, uint32_t v3, int32_t *adjTriangle) {
        // Check not connecting with self and not going outside the contour
        return v1 != v2 && getAdjTriangleEntry(adjTriangle, v1, v2) != -1;
    }
};

struct checkSecondParticleFunctor {
    __device__
    bool operator()(uint32_t v1, uint32_t v2, uint32_t v3, int32_t *adjTriangle) {
        // No additional conditions
        return true;
    }
};

struct checkUniqueClusterParticleFunctor {
    __device__
    bool operator()(uint32_t v1, uint32_t c2, uint32_t c3, int32_t *clusterInds) {
        // Check not connecting with particle from two given clusters
        return clusterInds[v1] != -1 && clusterInds[v1] != c2 && clusterInds[v1] != c3;
    }
};

struct checkSpecificClusterParticleFunctor {
    __device__
    bool operator()(uint32_t v1, uint32_t c2, uint32_t c3, int32_t *clusterInds) {
        // Check connecting with particle from c2
        return clusterInds[v1] == c2;
    }
};

struct checkSameClusterParticleFunctor {
    __device__
    bool operator()(uint32_t v1, uint32_t v2, uint32_t c3, int32_t *clusterInds) {
        // Check connecting with same cluster and not with c3
        return clusterInds[v1] == clusterInds[v2] && clusterInds[v1] != c3;
    }
};

// Compare distances to all particles in a given cell
template <typename F1, typename F2>
__device__
void compareClosest(uint32_t *first_ind,            // output: closest particle found or numParticles
                    uint32_t *second_ind,           // output: 2nd closest particle found or numParticles
                    float *first_dist_sq,           // output: squared distance to the closest particle found
                    float *second_dist_sq,          // output: squared distance to 2nd closest particle found
                    int3 gridPos,                   // input: position of this cell in the grid
                    uint32_t index,                 // input: index of this particle / cluster
                    float3 pos,                     // input: position of this particle / cluster centroid
                    float3 *sortedPos,              // input: sorted particle positions
                    int32_t *condArray,             // input: helper array for search conditions
                    uint32_t condValue,             // input: helper value for search conditions
                    F1 generalCond,                 // input: functor that checks each new candidate particle
                    F2 secondFoundCond,             // input: functor that checks candidates for 2nd best particle
                    uint32_t *gridParticleIndex,    // input: sorted particle indices
                    uint32_t *cellStart,            // input: cell start index
                    uint32_t *cellEnd) {            // input: cell end index
    uint32_t gridHash = calcGridHash(gridPos);

    // Get start of bucket for this cell
    uint32_t startIndex = cellStart[gridHash];

    if (startIndex != 0xffffffff) // Cell is not empty
    {
        // Iterate over particles in this cell
        uint32_t endIndex = cellEnd[gridHash];

        for (uint32_t j = startIndex; j < endIndex; j++) {
            uint32_t v1 = gridParticleIndex[j]; // Real particle index
            if (generalCond(v1, index, condValue, condArray)) {
                float3 pos2 = sortedPos[j];
                float dist_sq = dot(pos - pos2, pos - pos2);
                if (dist_sq < *first_dist_sq) {
                    *first_dist_sq = dist_sq;
                    *first_ind = v1;
                }
                else if (v1 != *first_ind && dist_sq < *second_dist_sq) {
                    if (secondFoundCond(v1, *first_ind, condValue, condArray)) {
                        *second_dist_sq = dist_sq;
                        *second_ind = v1;
                    }
                }
            }
        }
    }
}

// Compare particle pairs within specified search range
template <typename F1, typename F2>
__device__
void connectPairs(uint32_t *first_ind,          // output: closest particle found or numParticles
                  uint32_t *second_ind,         // output: 2nd closest particle found or numParticles
                  float3 *startPos,             // input: positions of particles or clusters centroids
                  float3 *sortedPos,            // input: sorted particle positions
                  uint32_t index,               // input: index of this particle / cluster
                  uint32_t sortedIndex,         // input: sorted index of this particle / cluster
                  float minSearchRadius,        // input: min search radius for particles
                  float maxSearchRadius,        // input: max search radius for particles
                  int32_t *condArray,           // input: helper array for search conditions
                  uint32_t condValue,           // input: helper value for search conditions
                  F1 generalCond,               // input: functor that checks each new candidate particle
                  F2 secondFoundCond,           // input: functor that checks candidates for 2nd best particle
                  uint32_t *gridParticleIndex,  // input: sorted particle indices
                  uint32_t *cellStart,          // input: cell start index
                  uint32_t *cellEnd) {          // input: cell end index
    // Read particle / cluster data from sorted arrays
    float3 pos = startPos[index];

    // Get address in grid
    int3 gridPos = calcGridPos(pos);

    int32_t minSearchRadiusInCellsNumber = (int32_t)ceilf((minSearchRadius - 1.0f) / 2.0f);
    int32_t maxSearchRadiusInCellsNumber = (int32_t)ceilf((maxSearchRadius - 1.0f) / 2.0f);
    float first_dist_sq = maxSearchRadius * maxSearchRadius;
    float second_dist_sq = maxSearchRadius * maxSearchRadius;

    // Examine neighbouring cells between two radii
    for (int32_t z = -maxSearchRadiusInCellsNumber; z < -minSearchRadiusInCellsNumber; z++) {
        for (int32_t y = -maxSearchRadiusInCellsNumber; y <= maxSearchRadiusInCellsNumber; y++) {
            for (int32_t x = -maxSearchRadiusInCellsNumber; x <= maxSearchRadiusInCellsNumber; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                compareClosest(first_ind, second_ind, &first_dist_sq, &second_dist_sq,
                               neighbourPos, sortedIndex, pos, sortedPos, condArray,
                               condValue, generalCond, secondFoundCond,
                               gridParticleIndex, cellStart, cellEnd);
            }
        }
    }
    for (int32_t z = -minSearchRadiusInCellsNumber; z < minSearchRadiusInCellsNumber; z++) {
        for (int32_t y = -maxSearchRadiusInCellsNumber; y < -minSearchRadiusInCellsNumber; y++) {
            for (int32_t x = -maxSearchRadiusInCellsNumber; x <= maxSearchRadiusInCellsNumber; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                compareClosest(first_ind, second_ind, &first_dist_sq, &second_dist_sq,
                               neighbourPos, sortedIndex, pos, sortedPos, condArray,
                               condValue, generalCond, secondFoundCond,
                               gridParticleIndex, cellStart, cellEnd);
            }
        }
        for (int32_t y = -minSearchRadiusInCellsNumber; y < minSearchRadiusInCellsNumber; y++) {
            for (int32_t x = -maxSearchRadiusInCellsNumber; x < -minSearchRadiusInCellsNumber; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                compareClosest(first_ind, second_ind, &first_dist_sq, &second_dist_sq,
                               neighbourPos, sortedIndex, pos, sortedPos, condArray,
                               condValue, generalCond, secondFoundCond,
                               gridParticleIndex, cellStart, cellEnd);
            }
            for (int32_t x = minSearchRadiusInCellsNumber + 1; x <= maxSearchRadiusInCellsNumber; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                compareClosest(first_ind, second_ind, &first_dist_sq, &second_dist_sq,
                               neighbourPos, sortedIndex, pos, sortedPos, condArray,
                               condValue, generalCond, secondFoundCond,
                               gridParticleIndex, cellStart, cellEnd);
            }
        }
        for (int32_t y = minSearchRadiusInCellsNumber; y <= maxSearchRadiusInCellsNumber; y++) {
            for (int32_t x = -maxSearchRadiusInCellsNumber; x <= maxSearchRadiusInCellsNumber; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                compareClosest(first_ind, second_ind, &first_dist_sq, &second_dist_sq,
                               neighbourPos, sortedIndex, pos, sortedPos, condArray,
                               condValue, generalCond, secondFoundCond,
                               gridParticleIndex, cellStart, cellEnd);
            }
        }
    }
    for (int32_t z = minSearchRadiusInCellsNumber; z <= maxSearchRadiusInCellsNumber; z++) {
        for (int32_t y = -maxSearchRadiusInCellsNumber; y <= maxSearchRadiusInCellsNumber; y++) {
            for (int32_t x = -maxSearchRadiusInCellsNumber; x <= maxSearchRadiusInCellsNumber; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                compareClosest(first_ind, second_ind, &first_dist_sq, &second_dist_sq,
                               neighbourPos, sortedIndex, pos, sortedPos, condArray,
                               condValue, generalCond, secondFoundCond,
                               gridParticleIndex, cellStart, cellEnd);
            }
        }
    }
}

// Find and connect particle with two closest neighbours
__global__
void connectParticlesD(int32_t *adjTriangle,                // output: adjacency triangle
                       float3 *sortedPos,                   // input: sorted positions
                       uint32_t *gridParticleIndex,         // input: sorted particle indices
                       uint32_t *cellStart,                 // input: cell start index
                       uint32_t *cellEnd) {                 // input: cell end index
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= params.numParticles) return;

    uint32_t sortedIndex = gridParticleIndex[index]; // Original particle index

    uint32_t first_ind = params.numParticles;
    uint32_t second_ind = params.numParticles;

    connectPairs(&first_ind, &second_ind, sortedPos, sortedPos,
                 index, sortedIndex, 0.0f, params.searchRadius,
                 adjTriangle, params.numParticles,
                 checkParticleParticleFunctor(), checkSecondParticleFunctor(),
                 gridParticleIndex, cellStart, cellEnd);

    // Write new pairs
    if (first_ind < params.numParticles) {
        setAdjTriangleEntry(adjTriangle, sortedIndex, first_ind, 1);
    }
    if (second_ind < params.numParticles) {
        setAdjTriangleEntry(adjTriangle, sortedIndex, second_ind, 1);
    }
}

// Choose shortest distance between two particle pairs from two clusters
__device__
void chooseShortestEdge(uint32_t *first_ind,     // output: first edge vertex
                        uint32_t *second_ind,    // output: second edge vertex
                        uint32_t v1c1,           // input: 1st vertex from 1st cluster
                        uint32_t v2c1,           // input: 2nd vertex from 1st cluster or numParticles if none
                        uint32_t v1c2,           // input: 1st vertex from 2nd cluster
                        uint32_t v2c2,           // input: 2nd vertex from 2nd cluster or numParticles if none
                        float3 *oldPos) {        // input: particle positions
    float3 pos1 = oldPos[v1c1];
    float3 pos3 = oldPos[v1c2];
    float dist13sq = dot(pos1 - pos3, pos1 - pos3);

    float dist = dist13sq;
    *first_ind = v1c1;
    *second_ind = v1c2;

    if (v2c1 < params.numParticles) {
        float3 pos2 = oldPos[v2c1];
        float dist23sq = dot(pos2 - pos3, pos2 - pos3);
        if (dist > dist23sq) {
            dist = dist23sq;
            *first_ind = v2c1;
            *second_ind = v1c2;
        }
    }
    if (v2c2 < params.numParticles) {
        float3 pos4 = oldPos[v2c2];
        float dist14sq = dot(pos1 - pos4, pos1 - pos4);
        if (dist > dist14sq) {
            dist = dist14sq;
            *first_ind = v1c1;
            *second_ind = v2c2;
        }
        if (v2c1 < params.numParticles) {
            float3 pos2 = oldPos[v2c1];
            float dist24sq = dot(pos2 - pos4, pos2 - pos4);
            if (dist > dist24sq) {
                dist = dist24sq;
                *first_ind = v2c1;
                *second_ind = v2c2;
            }
        }
    }
}

// Connect this cluster with another one with (approximately) shortest edge possible
__device__
void connectClustersParticles(int32_t *adjTriangle,         // output: adjacency triangle
                              uint32_t *clusterEdges,       // output: edges connecting clusters
                              uint32_t first_ind,           // input: 1st vertex from 2nd cluster
                              uint32_t second_ind,          // input: 2nd vertex from 2nd cluster or numParticles
                              uint32_t offset,              // input: edge offset in clusterEdges (0 or 2)
                              uint32_t thisClusterInd,      // input: index of this cluster
                              uint32_t thatClusterInd,      // input: index of 2nd cluster
                              float minSearchRadius,        // input: min search radius for particles
                              float maxSearchRadius,        // input: max search radius for particles
                              float3 *centroids,            // input: clusters centroids
                              int32_t *clusterInds,         // input: cluster index for each vertex
                              float3 *sortedPos,            // input: sorted particle positions
                              float3 *oldPos,               // input: particle positions
                              uint32_t *gridParticleIndex,  // input: sorted particle indices
                              uint32_t *cellStart,          // input: cell start index
                              uint32_t *cellEnd) {          // input: cell end index
    uint32_t v1c2 = first_ind;
    uint32_t v2c2 = second_ind;
    first_ind = params.numParticles;
    second_ind = params.numParticles;

    // Find 2 particles from this cluster closest to centroid of that cluster
    while (first_ind == params.numParticles) {
        connectPairs(&first_ind, &second_ind, centroids, sortedPos,
                     thatClusterInd, thisClusterInd, minSearchRadius, maxSearchRadius,
                     clusterInds, thatClusterInd,
                     checkSpecificClusterParticleFunctor(), checkSameClusterParticleFunctor(),
                     gridParticleIndex, cellStart, cellEnd);
        minSearchRadius = maxSearchRadius;
        maxSearchRadius += params.clustersSearchRadius;;
    }

    uint32_t v1c1 = first_ind;
    uint32_t v2c1 = second_ind;

    // Choose shortest edge from 4 between this and that cluster
    chooseShortestEdge(&first_ind, &second_ind, v1c1, v2c1, v1c2, v2c2, oldPos);

    // Save chosen edge
    if (first_ind < params.numParticles && second_ind < params.numParticles) {
        setAdjTriangleEntry(adjTriangle, first_ind, second_ind, 1);
    }
    clusterEdges[4 * thisClusterInd + offset] = first_ind;
    clusterEdges[4 * thisClusterInd + offset + 1] = second_ind;
}

// Connect this cluster with two closest neighbours
__global__
void connectClustersD(uint32_t *candidatesInds,         // output: pairs of particles from two neighbours
                      int32_t *adjTriangle,             // output: adjacency triangle
                      uint32_t *clusterEdges,           // output: edges connecting clusters
                      bool *clustersMerged,             // output: whether any cluster can be merged
                      uint32_t numClusters,             // input: total number of clusters
                      float minSearchRadius,            // input: min search radius for particles
                      float maxSearchRadius,            // input: max search radius for particles
                      float3 *centroids,                // input: clusters centroids
                      int32_t *clusterInds,             // input: cluster index for each vertex
                      float3 *sortedPos,                // input: sorted particle positions
                      float3 *oldPos,                   // input: particle positions
                      uint32_t *gridParticleIndex,      // input: sorted particle indices
                      uint32_t *cellStart,              // input: cell start index
                      uint32_t *cellEnd) {              // input: cell end index
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numClusters) return;

    uint32_t first_ind = params.numParticles;
    uint32_t second_ind = params.numParticles;

    // Find 2 candidates from the another cluster
    connectPairs(&first_ind, &second_ind, centroids, sortedPos,
                 index, index, minSearchRadius, maxSearchRadius,
                 clusterInds, params.numParticles,
                 checkUniqueClusterParticleFunctor(), checkSameClusterParticleFunctor(),
                 gridParticleIndex, cellStart, cellEnd);

    // Write new pairs
    candidatesInds[4 * index] = first_ind;
    candidatesInds[4 * index + 1] = second_ind;

    if (first_ind < params.numParticles) {
        clustersMerged[0] = true;
        uint32_t clusterInd = clusterInds[first_ind];

        // Connect two clusters with shortest edge
        connectClustersParticles(adjTriangle, clusterEdges,
                                 first_ind, second_ind, 0, index, clusterInd, minSearchRadius,
                                 maxSearchRadius, centroids, clusterInds, sortedPos, oldPos,
                                 gridParticleIndex, cellStart, cellEnd);

        // Find 2 candidates from another cluster
        first_ind = params.numParticles;
        second_ind = params.numParticles;
        connectPairs(&first_ind, &second_ind, centroids, sortedPos,
                     index, index, minSearchRadius, maxSearchRadius,
                     clusterInds, clusterInd,
                     checkUniqueClusterParticleFunctor(), checkSameClusterParticleFunctor(),
                     gridParticleIndex, cellStart, cellEnd);

        // Write new pairs
        candidatesInds[4 * index + 2] = first_ind;
        candidatesInds[4 * index + 3] = second_ind;

        if (first_ind < params.numParticles) {
            clusterInd = clusterInds[first_ind];

            // Connect two clusters with shortest edge
            connectClustersParticles(adjTriangle, clusterEdges,
                                     first_ind, second_ind, 2, index, clusterInd, minSearchRadius,
                                     maxSearchRadius, centroids, clusterInds, sortedPos, oldPos,
                                     gridParticleIndex, cellStart, cellEnd);
        }

//        printf("[%d] Connect clusters. Candidates: %d %d %d %d, edges: (%d %d) (%d %d)\n",
//                index, candidatesInds[4 * index], candidatesInds[4 * index + 1], candidatesInds[4 * index + 2],
//                candidatesInds[4 * index + 3], clusterEdges[4 * index], clusterEdges[4 * index + 1],
//                clusterEdges[4 * index + 2], clusterEdges[4 * index + 3]);
    }
}

// Sum degrees and edges count
__global__
void calcDegreesD(uint32_t *edgesCount,                 // output: number of edges
                  uint32_t *degrees,                    // output: vertices degrees
                  int32_t *adjTriangle) {               // input: adjacency triangle
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= params.numParticles) return;

    degrees[index] = 0;
    for (int32_t i = 0; i < index; i++) {
        if (getAdjTriangleEntry(adjTriangle, i, index) > 0) {
            atomicAdd(edgesCount, 1);
            degrees[index] += 1;
        }
    }
    for (int32_t i = index + 1; i < params.numParticles; i++) {
        if (getAdjTriangleEntry(adjTriangle, index, i) > 0) {
            atomicAdd(edgesCount, 1);
            degrees[index] += 1;
        }
    }
}

// Find isolated vertices
__global__
void markIsolatedVerticesD(char *isolatedVertices,     // output: isolated vertices
                           uint32_t *degrees) {        // input: vertices degrees
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= params.numParticles) return;

    if (degrees[index] == 0) {
        isolatedVertices[index] = 1;
    }
    else {
        isolatedVertices[index] = 0;
    }
}

// Create adjacency list
__global__
void createAdjListD(uint32_t *adjacencyList,           // output: adjacency list
                    int32_t *adjTriangle,              // input: adjacency triangle
                    uint32_t *edgesOffset,             // input: scanned vertices degrees
                    uint32_t *edgesSize) {             // input: vertices degrees
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= params.numParticles) return;

    uint32_t currentPos = edgesOffset[index];
    for (uint32_t i = 0; i < index; i++) {
        if (getAdjTriangleEntry(adjTriangle, index, i) > 0) {
            if (currentPos >= edgesOffset[index] + edgesSize[index]) {
                printf("[%d] Create adj list: index out of bonds: %d %d %d!\n", index, currentPos,
                       edgesOffset[index], edgesSize[index]);
            }
            adjacencyList[currentPos] = i;
            currentPos++;
        }
    }
    for (uint32_t i = index + 1; i < params.numParticles; i++) {
        if (getAdjTriangleEntry(adjTriangle, index, i) > 0) {
            if (currentPos >= edgesOffset[index] + edgesSize[index]) {
                printf("[%d] Create adj list: index out of bonds: %d %d %d!\n", index, currentPos,
                       edgesOffset[index], edgesSize[index]);
            }
            adjacencyList[currentPos] = i;
            currentPos++;
        }
    }
}

// Print adjacency list content to stdout
__global__
void readAdjListD(uint32_t *adjacencyList,           // input: adjacency list
                  uint32_t *edgesOffset,             // input: scanned degrees
                  uint32_t *edgesSize) {             // input: vertices degrees
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= params.numParticles) return;

    for (uint32_t i = edgesOffset[index]; i < edgesOffset[index] + edgesSize[index]; i++) {
        printf("[%d] %d\n", index, adjacencyList[i]);
    }
}

#endif // __PARTICLES_KERNELS_H__