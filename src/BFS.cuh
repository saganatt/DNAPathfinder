#ifndef __BFS_H__
#define __BFS_H__

// Code adapted from: https://github.com/rafalk342/bfs-cuda

#include "cluster.h"
#include "helpers.cuh"

extern "C"
{
// Select vertices to process at the next stage    
__global__
void nextLayerD(float *edgesLengths,                   // output: edge lengths
                float *d_distance,                     // output: distance to each visited vertex
                uint32_t *d_verticesDistance,          // output: distance to each visited vertex in vertices
                char *d_frontier,                      // input: vertices to process at the next stage
                int32_t *clusterInds,                  // output: cluster index for each vertex
                float3 *pos,                           // input: vertices positions
                uint32_t *d_adjacencyList,             // input: adjacency list
                uint32_t *d_edgesOffset,               // input: vertices scanned degrees
                uint32_t *d_edgesSize,                 // input: vertices degrees
                uint32_t *d_currentQueue,              // input: vertices processed at this stage
                int32_t level,                         // input: current BFS level
                uint32_t currentClusterInd,            // input: index of current cluster
                uint32_t queueSize) {                  // input: number of vertices processed at this stage
    uint32_t thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        uint32_t u = d_currentQueue[thid];
        for (int32_t i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            uint32_t v = d_adjacencyList[i];
            float dist = sqrtf(dot(pos[v] - pos[u], pos[v] - pos[u]));
            edgesLengths[i] = dist;
            atomicMinf(&d_distance[v], d_distance[u] + dist);
            if (level + 1 < d_verticesDistance[v]) {
                d_verticesDistance[v] = level + 1;
                d_frontier[v] = 1;
            }
            clusterInds[v] = currentClusterInd;
        }
    }
}

// Sum degrees of vertices at BFS frontier, count only edges to the next frontier
__global__
void countDegreesD(uint32_t *d_degrees,                // output: selected vertices degrees
                   char *d_frontier,                   // input: vertices to process at the next stage
                   uint32_t *d_adjacencyList,          // input: adjacency list
                   uint32_t *d_edgesOffset,            // input: vertices scanned degrees
                   uint32_t *d_edgesSize,              // input: vertices degrees
                   uint32_t *d_currentQueue,           // input: vertices processed at this stage
                   uint32_t queueSize) {               // input: number of vertices processed at this stage
    uint32_t thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        uint32_t u = d_currentQueue[thid];
        uint32_t degree = 0;
        for (int32_t i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            uint32_t v = d_adjacencyList[i];
            if (d_frontier[v] == 1) {
                ++degree;
            }
        }
        d_degrees[thid] = degree;
    }
}

// Sum degrees of vertices at BFS frontier, count only edges to the next frontier
// Avoid repeated vertices in the queue
__global__
void countDegrees2D(uint32_t *d_degrees,                // output: selected vertices degrees
                    char *d_frontier,                   // input: vertices to process at the next stage
                    uint32_t *d_adjacencyList,          // input: adjacency list
                    uint32_t *d_edgesOffset,            // input: vertices scanned degrees
                    uint32_t *d_edgesSize,              // input: vertices degrees
                    uint32_t verticesCount) {           // input: total number of vertices in the graph
    uint32_t thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < verticesCount) {
        uint32_t degree = 0;
        for (int32_t i = d_edgesOffset[thid]; i < d_edgesOffset[thid] + d_edgesSize[thid]; i++) {
            uint32_t v = d_adjacencyList[i];
            if (d_frontier[v] == 1) {
                ++degree;
            }
        }
        d_degrees[thid] = degree;
    }
}

// Scan partial degrees
__global__
void scanDegreesD(uint32_t *d_degrees,                      // output: scanned degrees
                  uint32_t *d_incrDegrees,                  // output: array of degree scan block results
                  uint32_t size) {                          // input: size of array to scan
    uint32_t thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < size) {
        // Write initial values to shared memory
        __shared__ uint32_t prefixSum[64];
        uint32_t modulo = threadIdx.x;
        prefixSum[modulo] = d_degrees[thid];
        __syncthreads();

        // Calculate scan on this block
        // Go up
        for (uint32_t nodeSize = 2; nodeSize <= 64; nodeSize <<= 1) {
            if ((modulo & (nodeSize - 1)) == 0) {
                if (thid + (nodeSize >> 1) < size) {
                    uint32_t nextPosition = modulo + (nodeSize >> 1);
                    prefixSum[modulo] += prefixSum[nextPosition];
                }
            }
            __syncthreads();
        }

        // Write information for increment prefix sums
        if (modulo == 0) {
            uint32_t block = thid >> 6;
            d_incrDegrees[block + 1] = prefixSum[modulo];
        }

        // Go down
        for (uint32_t nodeSize = 64; nodeSize > 1; nodeSize >>= 1) {
            if ((modulo & (nodeSize - 1)) == 0) {
                if (thid + (nodeSize >> 1) < size) {
                    uint32_t next_position = modulo + (nodeSize >> 1);
                    uint32_t tmp = prefixSum[modulo];
                    prefixSum[modulo] -= prefixSum[next_position];
                    prefixSum[next_position] = tmp;

                }
            }
            __syncthreads();
        }
        d_degrees[thid] = prefixSum[modulo];
    }
}

// Create queue for next BFS stage
__global__
void assignVerticesNextQueueD(uint32_t *d_nextQueue,      // output: vertices processed at the next stage
                              uint32_t *d_adjacencyList,  // input: adjacency list
                              uint32_t *d_edgesOffset,    // input: vertices scanned degrees
                              uint32_t *d_edgesSize,      // input: vertices degrees
                              uint32_t *d_currentQueue,   // input: vertices processed at this stage
                              uint32_t *d_degrees,        // input: selected vertices degrees
                              uint32_t *d_incrDegrees,    // input: array of degrees scan block results
                              char *d_frontier,           // input: vertices to process at the next stage
                              uint32_t queueSize) {       // input: number of vertices processed at this stage
    uint32_t thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        __shared__ uint32_t sharedIncrement;
        if (!threadIdx.x) {
            sharedIncrement = d_incrDegrees[thid >> 6];
        }
        __syncthreads();

        uint32_t sum = 0;
        if (threadIdx.x) {
            sum = d_degrees[thid - 1];
        }

        uint32_t u = d_currentQueue[thid];
        int32_t counter = 0;
        for (int32_t i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            uint32_t v = d_adjacencyList[i];
            if (d_frontier[v] == 1) {
                int32_t nextQueuePlace = sharedIncrement + sum + counter;
                d_nextQueue[nextQueuePlace] = v;
                counter++;
            }
        }
    }
}

// Create queue for next BFS stage
// Avoid repeated vertices in the current queue
__global__
void assignVerticesNextQueue2D(uint32_t *d_nextQueue,       // output: vertices processed at the next stage
                               uint32_t *d_adjacencyList,   // input: adjacency list
                               uint32_t *d_edgesOffset,     // input: vertices scanned degrees
                               uint32_t *d_edgesSize,       // input: vertices degrees
                               uint32_t *d_degrees,         // input: selected vertices degrees
                               uint32_t *d_incrDegrees,     // input: array of degrees scan block results
                               char *d_frontier,            // input: vertices to process at the next stage
                               uint32_t verticesCount) {    // input: total number of vertices in the graph
    uint32_t thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < verticesCount) {
        __shared__ uint32_t sharedIncrement;
        if (!threadIdx.x) {
            sharedIncrement = d_incrDegrees[thid >> 6];
        }
        __syncthreads();

        uint32_t sum = 0;
        if (threadIdx.x) {
            sum = d_degrees[thid - 1];
        }

        int32_t counter = 0;
        for (int32_t i = d_edgesOffset[thid]; i < d_edgesOffset[thid] + d_edgesSize[thid]; i++) {
            uint32_t v = d_adjacencyList[i];
            if (d_frontier[v] == 1) {
                int32_t nextQueuePlace = sharedIncrement + sum + counter;
                d_nextQueue[nextQueuePlace] = v;
                counter++;
            }
        }
    }
}

}

#endif // __BFS_H__