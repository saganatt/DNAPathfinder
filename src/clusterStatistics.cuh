#ifndef __CLUSTER_STATISTICS_H__
#define __CLUSTER_STATISTICS_H__

#include <stdio.h>
#include <math.h>

#include "kernelParams.cuh"
#include "helpers.cuh"
#include "cluster.h"

// Maximum and minimum possible values for a particle coordinate
#define MAX_LENGTH 10000
#define MIN_LENGTH 0

// Helper function for reduce centroid calculations, adjusted from Mark Harris' code at:
// http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <uint32_t blockSize>
__device__
void warpReduceCentroid(volatile uint32_t *sharedClusterSize,     // output: shared cluster size
                        volatile float3 *sharedCentroid,          // output: shared coordinates sum for centroid
                        uint32_t tid) {                           // input: thread ID
    if (blockSize >= 64) {
        sharedClusterSize[tid] += sharedClusterSize[tid + 32];
        sharedCentroid[tid] += sharedCentroid[tid + 32];
    }
    if (blockSize >= 32) {
        sharedClusterSize[tid] += sharedClusterSize[tid + 16];
        sharedCentroid[tid] += sharedCentroid[tid + 16];
    }
    if (blockSize >= 16) {
        sharedClusterSize[tid] += sharedClusterSize[tid + 8];
        sharedCentroid[tid] += sharedCentroid[tid + 8];
    }
    if(blockSize >= 8) {
        sharedClusterSize[tid] += sharedClusterSize[tid + 4];
        sharedCentroid[tid] += sharedCentroid[tid + 4];
    }
    if (blockSize >= 4) {
        sharedClusterSize[tid] += sharedClusterSize[tid + 2];
        sharedCentroid[tid] += sharedCentroid[tid + 2];
    }
    if (blockSize >= 2) {
        sharedClusterSize[tid] += sharedClusterSize[tid + 1];
        sharedCentroid[tid] += sharedCentroid[tid + 1];
    }
}

// Fill in cluster centroid based on BFS results
// Code adjusted from Mark Harris' code at:
// http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// and from NVIDIA reduction sample.
template <uint32_t blockSize, bool nIsPow2>
__global__
void calcClusterCentroidD(Cluster *cluster,                    // output: current cluster
                          float3 *oldPos,                      // input: particle positions
                          int32_t *clusterInds,                // input: cluster index for each vertex
                          uint32_t currentClusterInd) {        // input: index of current cluster
    extern __shared__ float3 sharedCentroidArray[];

    uint32_t sharedMemoryMul = (blockDim.x <= 32) ? 2 * blockDim.x : blockDim.x;
    float3 *sharedCentroid = (float3*)sharedCentroidArray;
    uint32_t *sharedClusterSize = (uint32_t*)&sharedCentroidArray[sharedMemoryMul];

    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * (blockSize * 2) + tid;
    uint32_t gridSize = blockSize * 2 * gridDim.x;

    sharedClusterSize[tid] = 0;
    sharedCentroid[tid] = make_float3(0.0f);

    while (i < params.numParticles) {
        if(clusterInds[i] == currentClusterInd) {
            sharedClusterSize[tid] += 1;
            sharedCentroid[tid] += oldPos[i];
        }
        if (nIsPow2 || i + blockSize < params.numParticles) {
            if(clusterInds[i + blockSize] == currentClusterInd) {
                sharedClusterSize[tid] += 1;
                sharedCentroid[tid] += oldPos[i + blockSize];
            }
        }
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 1024) {
        if (tid < 512) {
            sharedClusterSize[tid] += sharedClusterSize[tid + 512];
            sharedCentroid[tid] += sharedCentroid[tid + 512];
        }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) {
            sharedClusterSize[tid] += sharedClusterSize[tid + 256];
            sharedCentroid[tid] += sharedCentroid[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sharedClusterSize[tid] += sharedClusterSize[tid + 128];
            sharedCentroid[tid] += sharedCentroid[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid <  64) {
            sharedClusterSize[tid] += sharedClusterSize[tid + 64];
            sharedCentroid[tid] += sharedCentroid[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32) {
        warpReduceCentroid<blockSize>(sharedClusterSize, sharedCentroid, tid);
    }
    if (tid == 0) {
        atomicAdd(&(cluster->clusterSize), sharedClusterSize[0]);
        atomicAdd(&(cluster->centroid.x), sharedCentroid[0].x);
        atomicAdd(&(cluster->centroid.y), sharedCentroid[0].y);
        atomicAdd(&(cluster->centroid.z), sharedCentroid[0].z);
    }
}

// Compare edge lengths and store the results in shared arrays
__device__
void compareAndReduceEdgeLengths(volatile float *sharedMinEdge,            // output: shared min edges
                                 volatile float *sharedMaxEdge,            // output: shared max edges
                                 uint32_t sind,                            // input: shared arrays index
                                 uint32_t ind) {                           // input: input arrays index
    if (sharedMinEdge[ind] < sharedMinEdge[sind]) {
        sharedMinEdge[sind] = sharedMinEdge[ind];
    }
    if (sharedMaxEdge[ind] > sharedMaxEdge[sind]) {
        sharedMaxEdge[sind] = sharedMaxEdge[ind];
    }
}

// Helper function for reduce edge lengths calculations, adjusted from Mark Harris' code at:
// http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <uint32_t blockSize>
__device__
void warpReduceEdgeLengths(volatile float *sharedMinEdge,            // output: shared min edges
                           volatile float *sharedMaxEdge,            // output: shared max edges
                           uint32_t tid) {                           // input: thread ID
    if (blockSize >= 64) {
        compareAndReduceEdgeLengths(sharedMinEdge, sharedMaxEdge, tid, tid + 32);
    }
    if (blockSize >= 32) {
        compareAndReduceEdgeLengths(sharedMinEdge, sharedMaxEdge, tid, tid + 16);
    }
    if (blockSize >= 16) {
        compareAndReduceEdgeLengths(sharedMinEdge, sharedMaxEdge, tid, tid + 8);
    }
    if(blockSize >= 8) {
        compareAndReduceEdgeLengths(sharedMinEdge, sharedMaxEdge, tid, tid + 4);
    }
    if (blockSize >= 4) {
        compareAndReduceEdgeLengths(sharedMinEdge, sharedMaxEdge, tid, tid + 2);
    }
    if (blockSize >= 2) {
        compareAndReduceEdgeLengths(sharedMinEdge, sharedMaxEdge, tid, tid + 1);
    }
}

// Fill in cluster edge lengths based on BFS results
// Code adjusted from Mark Harris' code at:
// http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// and from NVIDIA reduction sample.
template <uint32_t blockSize, bool nIsPow2>
__global__
void calcClusterEdgeLengthsD(Cluster *cluster,                    // output: current cluster
                             float *edgesLengths,                 // input: edges lengths, -1 if outside the cluster
                             uint32_t edgesCount) {               // input: number of edges)
    extern __shared__ float sharedEdgeArray[];

    uint32_t sharedMemoryMul = (blockDim.x <= 32) ? 2 * blockDim.x : blockDim.x;
    float *sharedMinEdge = sharedEdgeArray;
    float *sharedMaxEdge = &sharedEdgeArray[sharedMemoryMul];

    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * (blockSize * 2) + tid;
    uint32_t gridSize = blockSize * 2 * gridDim.x;

    sharedMinEdge[tid] = MAX_LENGTH;
    sharedMaxEdge[tid] = MIN_LENGTH;

    while (i < edgesCount) {
        if (edgesLengths[i] >= 0.0f && edgesLengths[i] < sharedMinEdge[tid]) {
            sharedMinEdge[tid] = edgesLengths[i];
        }
        if (edgesLengths[i] > sharedMaxEdge[tid]) {
            sharedMaxEdge[tid] = edgesLengths[i];
        }
        if (nIsPow2 || i + blockSize < edgesCount) {
            if (edgesLengths[i + blockSize] >= 0.0f
                && edgesLengths[i + blockSize] < sharedMinEdge[tid]) {
                sharedMinEdge[tid] = edgesLengths[i + blockSize];
            }
            if (edgesLengths[i + blockSize] > sharedMaxEdge[tid]) {
                sharedMaxEdge[tid] = edgesLengths[i + blockSize];
            }
        }
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 1024) {
        if (tid < 512) {
            compareAndReduceEdgeLengths(sharedMinEdge, sharedMaxEdge, tid, tid + 512);
        }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) {
            compareAndReduceEdgeLengths(sharedMinEdge, sharedMaxEdge, tid, tid + 256);
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            compareAndReduceEdgeLengths(sharedMinEdge, sharedMaxEdge, tid, tid + 128);
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid <  64) {
            compareAndReduceEdgeLengths(sharedMinEdge, sharedMaxEdge, tid, tid + 64);
        }
        __syncthreads();
    }
    if (tid < 32) {
        warpReduceEdgeLengths<blockSize>(sharedMinEdge, sharedMaxEdge, tid);
    }
    if (tid == 0) {
        atomicMinf(&(cluster->shortestEdge), sharedMinEdge[0]);
        atomicMaxf(&(cluster->longestEdge), sharedMaxEdge[0]);
    }
}

// Compare path lengths and store the results in shared arrays
__device__
void compareAndReducePathLengths(volatile float *sharedMaxPath,             // output: shared max path length
                                 volatile uint32_t *sharedMaxPathVertices,  // output: shared max path length in
                                                                            //         vertices
                                 uint32_t sind,                             // input: shared arrays index
                                 uint32_t ind) {                            // input: input arrays index
    if (sharedMaxPath[ind] > sharedMaxPath[sind]) {
        sharedMaxPath[sind] = sharedMaxPath[ind];
    }
    if (sharedMaxPathVertices[ind] > sharedMaxPathVertices[sind]) {
        sharedMaxPathVertices[sind] = sharedMaxPathVertices[ind];
    }
}

// Helper function for reduce path lengths calculations, adjusted from Mark Harris' code at:
// http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <uint32_t blockSize>
__device__
void warpReducePathLengths(volatile float *sharedMaxPath,             // output: shared max path length
                           volatile uint32_t *sharedMaxPathVertices,  // output: shared max path length in vertices
                           uint32_t tid) {                            // input: thread ID
    if (blockSize >= 64) {
        compareAndReducePathLengths(sharedMaxPath, sharedMaxPathVertices, tid, tid + 32);
    }
    if (blockSize >= 32) {
        compareAndReducePathLengths(sharedMaxPath, sharedMaxPathVertices, tid, tid + 16);
    }
    if (blockSize >= 16) {
        compareAndReducePathLengths(sharedMaxPath, sharedMaxPathVertices, tid, tid + 8);
    }
    if(blockSize >= 8) {
        compareAndReducePathLengths(sharedMaxPath, sharedMaxPathVertices, tid, tid + 4);
    }
    if (blockSize >= 4) {
        compareAndReducePathLengths(sharedMaxPath, sharedMaxPathVertices, tid, tid + 2);
    }
    if (blockSize >= 2) {
        compareAndReducePathLengths(sharedMaxPath, sharedMaxPathVertices, tid, tid + 1);
    }
}

// Fill in cluster path lengths based on BFS results
// Code adjusted from Mark Harris' code at:
// http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// and from NVIDIA reduction sample.
template <uint32_t blockSize, bool nIsPow2>
__global__
void calcClusterPathLengthsD(Cluster *cluster,                    // output: current cluster
                             float *distance,                     // input: distance to each vertex
                             uint32_t *verticesDistance,          // input: distance to each vertex in vertices
                             int32_t *clusterInds,                // input: cluster index for each vertex
                             uint32_t currentClusterInd) {        // input: index of current cluster
    extern __shared__ float sharedPathArray[];

    uint32_t sharedMemoryMul = (blockDim.x <= 32) ? 2 * blockDim.x : blockDim.x;
    float *sharedMaxPath = sharedPathArray;
    uint32_t *sharedMaxPathVertices = (uint32_t*)&sharedPathArray[sharedMemoryMul];

    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * (blockSize * 2) + tid;
    uint32_t gridSize = blockSize * 2 * gridDim.x;

    sharedMaxPath[tid] = MIN_LENGTH;
    sharedMaxPathVertices[tid] = MIN_LENGTH;

    while (i < params.numParticles) {
        if(clusterInds[i] == currentClusterInd) {
            if (distance[i] > sharedMaxPath[tid]) {
                sharedMaxPath[tid] = distance[i];
            }
            if (verticesDistance[i] > sharedMaxPathVertices[tid]) {
                sharedMaxPathVertices[tid] = verticesDistance[i];
            }
        }
        if (nIsPow2 || i + blockSize < params.numParticles) {
            if(clusterInds[i + blockSize] == currentClusterInd) {
                if (distance[i + blockSize] > sharedMaxPath[tid]) {
                    sharedMaxPath[tid] = distance[i + blockSize];
                }
                if (verticesDistance[i + blockSize] > sharedMaxPathVertices[tid]) {
                    sharedMaxPathVertices[tid] = verticesDistance[i + blockSize];
                }
            }
        }
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 1024) {
        if (tid < 512) {
            compareAndReducePathLengths(sharedMaxPath, sharedMaxPathVertices, tid, tid + 512);
        }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) {
            compareAndReducePathLengths(sharedMaxPath, sharedMaxPathVertices, tid, tid + 256);
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            compareAndReducePathLengths(sharedMaxPath, sharedMaxPathVertices, tid, tid + 128);
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid <  64) {
            compareAndReducePathLengths(sharedMaxPath, sharedMaxPathVertices, tid, tid + 64);
        }
        __syncthreads();
    }
    if (tid < 32) {
        warpReducePathLengths<blockSize>(sharedMaxPath, sharedMaxPathVertices, tid);
    }
    if (tid == 0) {
        atomicMaxf(&(cluster->longestPath), sharedMaxPath[0]);
        atomicMax(&(cluster->longestPathVertices), sharedMaxPathVertices[0]);
    }
}

// Helper function for reduce branchings calculations, adjusted from Mark Harris' code at:
// http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <uint32_t blockSize>
__device__
void warpReduceBranchings(volatile uint32_t *sharedBranchingsCount,     // output: shared branchings count
                          uint32_t tid) {                               // input: thread ID
    if (blockSize >= 64) {
        sharedBranchingsCount[tid] += sharedBranchingsCount[tid + 32];
    }
    if (blockSize >= 32) {
        sharedBranchingsCount[tid] += sharedBranchingsCount[tid + 16];
    }
    if (blockSize >= 16) {
        sharedBranchingsCount[tid] += sharedBranchingsCount[tid + 8];
    }
    if(blockSize >= 8) {
        sharedBranchingsCount[tid] += sharedBranchingsCount[tid + 4];
    }
    if (blockSize >= 4) {
        sharedBranchingsCount[tid] += sharedBranchingsCount[tid + 2];
    }
    if (blockSize >= 2) {
        sharedBranchingsCount[tid] += sharedBranchingsCount[tid + 1];
    }
}

// Fill in cluster branchings count based on BFS results
// Code adjusted from Mark Harris' code at:
// http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// and from NVIDIA reduction sample.
template <uint32_t blockSize, bool nIsPow2>
__global__
void calcClusterBranchingsCountD(Cluster *cluster,             // output: current cluster
                          uint32_t *degrees,                   // output: vertices degrees
                          int32_t *clusterInds,                // input: cluster index for each vertex
                          uint32_t currentClusterInd) {        // input: index of current cluster
    extern __shared__ uint32_t sharedBranchingsCount[];

    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * (blockSize * 2) + tid;
    uint32_t gridSize = blockSize * 2 * gridDim.x;

    sharedBranchingsCount[tid] = 0;

    while (i < params.numParticles) {
        if(clusterInds[i] == currentClusterInd && degrees[i] > 2) {
            sharedBranchingsCount[tid] += 1;
        }
        if (nIsPow2 || i + blockSize < params.numParticles) {
            if(clusterInds[i + blockSize] == currentClusterInd && degrees[i + blockSize] > 2) {
                sharedBranchingsCount[tid] += 1;
            }
        }
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 1024) {
        if (tid < 512) {
            sharedBranchingsCount[tid] += sharedBranchingsCount[tid + 512];
        }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) {
            sharedBranchingsCount[tid] += sharedBranchingsCount[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sharedBranchingsCount[tid] += sharedBranchingsCount[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid <  64) {
            sharedBranchingsCount[tid] += sharedBranchingsCount[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32) {
        warpReduceBranchings<blockSize>(sharedBranchingsCount, tid);
    }
    if (tid == 0) {
        atomicAdd(&(cluster->branchingsCount), sharedBranchingsCount[0]);
    }
}

#endif // __CLUSTER_STATISTICS_H__