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

extern "C"
{
void cudaInit(int argc, char **argv);

void allocateArray(void **devPtr, int size);
void freeArray(void *devPtr);

void threadSync();

void copyArrayFromDevice(void *host, const void *device, int size);
void copyArrayToDevice(void *device, const void *host, int offset, int size);

void setParameters(SimParams *hostParams);

void calcHash(uint  *gridParticleHash,
              uint  *gridParticleIndex,
              float *pos,
              int    numParticles);

void reorderDataAndFindCellStart(uint  *cellStart,
                                 uint  *cellEnd,
                                 float *sortedPos,
                                 uint  *gridParticleHash,
                                 uint  *gridParticleIndex,
                                 float *oldPos,
                                 uint   numParticles,
                                 uint   numCells);

void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);

void checkContour(int32_t *adjTriangle,
                  uint32_t adjTriangleSize,
                  float *oldPos,
                  uint32_t *contour,
                  uint3 contourSize,
                  uint numParticles);

void connectPairs(int32_t *adjTriangle,
             float *sortedPos,
             uint *gridParticleIndex,
             uint *cellStart,
             uint *cellEnd,
             uint numParticles);

void calcDegrees(int32_t *adjTriangle,
                 int32_t *edgesCount,
                 int32_t *degrees,
                 uint numParticles);

void markIsolatedVertices(int32_t *degrees,
                          bool *isolatedVertices,
                          uint   numParticles);

void createAdjList(int32_t *adjacencyList,
                   int32_t *adjTriangle,
                   int32_t *edgesOffset,
                   int32_t *edgesSize,
                   uint numParticles);

void readAdjList(int32_t *adjacencyList,
                 int32_t *edgesOffset,
                 int32_t *edgesSize,
                 uint numParticles);

void nextLayer(int32_t level,
               int32_t *adjacencyList,
               int32_t *edgesOffset,
               int32_t *edgesSize,
               float *distance,
               int32_t *verticesDistance,
               int32_t *parent,
               int queueSize,
               int32_t *currentQueue,
               float *oldPos,
               bool *frontier,
               Cluster *cluster,
               int32_t *clusterInds,
               int32_t currentClusterInd);

void completeClusterStats(int32_t *edgesSize,
                          float *oldPos,
                          int numParticles,
                          bool *frontier,
                          Cluster *cluster);

void countDegrees(int32_t *adjacencyList,
                  int32_t *edgesOffset,
                  int32_t *edgesSize,
                  int32_t *parent,
                  int queueSize,
                  int32_t *currentQueue,
                  int32_t *degrees,
                  bool *frontier);

void scanDegreesTh(uint numParticles, int32_t *degrees, int32_t *scannedDegrees);
void scanDegrees(int queueSize, int32_t *degrees, int32_t *incrDegrees, int32_t *scannedDegrees);

void assignVerticesNextQueue(int32_t *adjacencyList,
                             int32_t *edgesOffset,
                             int32_t *edgesSize,
                             int32_t *parent,
                             int queueSize,
                             int32_t *currentQueue,
                             int32_t *nextQueue,
                             int32_t *degrees,
                             int32_t *incrDegrees,
                             int32_t nextQueueSize,
                             bool *frontier);

}