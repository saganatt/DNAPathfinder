// Code adapted from: https://github.com/rafalk342/bfs-cuda

#include <helper_functions.h>

#include "cluster.h"

extern "C"
{

__device__
void fillCluster(Cluster *cluster, float3 pos, int32_t ind, float edgeLength, int32_t vertDistance, float distance)
{
    if(pos.x < cluster->minExtremes.x) {
        cluster->minExtremes.x = pos.x;
        cluster->minExtremesInd.x = ind;
    }
    if(pos.y < cluster->minExtremes.y) {
        cluster->minExtremes.y = pos.y;
        cluster->minExtremesInd.y = ind;
    }
    if(pos.z < cluster->minExtremes.z) {
        cluster->minExtremes.z = pos.z;
        cluster->minExtremesInd.z = ind;
    }
    if(pos.x > cluster->maxExtremes.x) {
        cluster->maxExtremes.x = pos.x;
        cluster->maxExtremesInd.x = ind;
    }
    if(pos.y > cluster->maxExtremes.y) {
        cluster->maxExtremes.y = pos.y;
        cluster->maxExtremesInd.y = ind;
    }
    if(pos.z > cluster->maxExtremes.z) {
        cluster->maxExtremes.z = pos.z;
        cluster->maxExtremesInd.z = ind;
    }
    if(edgeLength > cluster->longestEdge) {
        cluster->longestEdge = edgeLength;
    }
    if(edgeLength < cluster->shortestEdge) {
        cluster->shortestEdge = edgeLength;
    }
    if(distance > cluster->longestPath) {
        cluster->longestPath = distance;
    }
    if(vertDistance > cluster->longestPathVertices) {
        cluster->longestPathVertices = vertDistance;
    }
}

__global__
void nextLayerD(int32_t level, int32_t* d_adjacencyList, int32_t* d_edgesOffset, int32_t* d_edgesSize,
                float* d_distance, int32_t *d_verticesDistance, int32_t* d_parent, int32_t queueSize,
                int32_t* d_currentQueue, float3 *pos, bool *frontier,
                Cluster *cluster, int32_t *clusterInds, int32_t currentClusterInd)
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int u = d_currentQueue[thid];
        for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            int v = d_adjacencyList[i];
            //printf("[%d] Next layer. Level %d, checking pair %d %d\n", thid, level, u, v);
            float dist = sqrtf(dot(pos[v] - pos[u], pos[v] - pos[u]));
            if (d_distance[u] + dist < d_distance[v]) {
                d_distance[v] = d_distance[u] + dist;
                //d_parent[v] = u;
            }
            if (level + 1 < d_verticesDistance[v]) {
                d_verticesDistance[v] = level + 1;
                d_parent[v] = i;
                frontier[v] = true;
                //printf("[%d] Next layer. Level %d, %d is on frontier\n", thid, level, v);
            }
            clusterInds[v] = currentClusterInd;
            fillCluster(cluster, pos[v], v, dist, d_verticesDistance[v], d_distance[v]);
        }
    }
}

__global__
void countDegreesD(int32_t* d_adjacencyList, int32_t* d_edgesOffset, int32_t* d_edgesSize, int32_t* d_parent,
                  int32_t queueSize, int32_t* d_currentQueue, int32_t* d_degrees, bool *frontier)
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int u = d_currentQueue[thid];
        int degree = 0;
        for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            int v = d_adjacencyList[i];
            if (d_parent[v] == i && v != u && frontier[v]) {
                ++degree;
            }
        }
        d_degrees[thid] = degree;
    }
}

__global__
void scanDegreesD(int32_t size, int32_t* d_degrees, int32_t* d_incrDegrees, int32_t *d_scannedDegrees)
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < size) {
        //write initial values to shared memory
        __shared__ int prefixSum[64];
        int modulo = threadIdx.x;
        prefixSum[modulo] = d_degrees[thid];
        __syncthreads();

        //calculate scan on this block
        //go up
        for (int nodeSize = 2; nodeSize <= 64; nodeSize <<= 1) {
            if ((modulo & (nodeSize - 1)) == 0) {
                if (thid + (nodeSize >> 1) < size) {
                    int nextPosition = modulo + (nodeSize >> 1);
                    prefixSum[modulo] += prefixSum[nextPosition];
                }
            }
            __syncthreads();
        }

        //write information for increment prefix sums
        if (modulo == 0) {
            int block = thid >> 6;
            d_incrDegrees[block + 1] = prefixSum[modulo];
        }

        //go down
        for (int nodeSize = 64; nodeSize > 1; nodeSize >>= 1) {
            if ((modulo & (nodeSize - 1)) == 0) {
                if (thid + (nodeSize >> 1) < size) {
                    int next_position = modulo + (nodeSize >> 1);
                    int tmp = prefixSum[modulo];
                    prefixSum[modulo] -= prefixSum[next_position];
                    prefixSum[next_position] = tmp;

                }
            }
            __syncthreads();
        }
        d_scannedDegrees[thid] = prefixSum[modulo];
    }

}

__global__
void assignVerticesNextQueueD(int32_t* d_adjacencyList, int32_t* d_edgesOffset, int32_t* d_edgesSize,
                              int32_t* d_parent, int32_t queueSize, int32_t* d_currentQueue, int32_t* d_nextQueue,
                              int32_t* d_degrees, int32_t* d_incrDegrees,
                              int32_t nextQueueSize, bool *frontier)
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        __shared__ int sharedIncrement;
        if (!threadIdx.x) {
            sharedIncrement = d_incrDegrees[thid >> 6];
        }
        __syncthreads();

        int sum = 0;
        if (threadIdx.x) {
            sum = d_degrees[thid - 1];
        }

        int u = d_currentQueue[thid];
        int counter = 0;
        for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            int v = d_adjacencyList[i];
            if (d_parent[v] == i && v != u && frontier[v]) {
                int nextQueuePlace = sharedIncrement + sum + counter;
                d_nextQueue[nextQueuePlace] = v;
                counter++;
            }
        }
    }
}

}