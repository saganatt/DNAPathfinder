#ifndef __KERNEL_WRAPPERS_H__
#define __KERNEL_WRAPPERS_H__

extern "C"
{
// Functions from NVIDIA's particles sample
void cudaInit(int32_t argc, char **argv);

void freeArray(void *devPtr);
void allocateArray(void **devPtr, size_t size);

void threadSync();

void setArray(void *devPtr, int32_t value, size_t count);
void copyArrayFromDevice(void *host, const void *device, int32_t size);
void copyArrayToDevice(void *device, const void *host, int32_t size);

void setParameters(KernelParams *hostParams);

void calcHash(uint32_t *gridParticleHash,
              uint32_t *gridParticleIndex,
              float3 *pos,
              uint32_t numParticles);

void reorderDataAndFindCellStart(uint32_t *cellStart,
                                 uint32_t *cellEnd,
                                 float3 *sortedPos,
                                 uint32_t *gridParticleHash,
                                 uint32_t *gridParticleIndex,
                                 float3 *oldPos,
                                 uint32_t numParticles);

// Custom functions
void sortParticles(uint32_t *dGridParticleHash, uint32_t *dGridParticleIndex, uint32_t numParticles);

void checkContour(int32_t *adjTriangle,
                  uint32_t adjTriangleSize,
                  float3 *oldPos,
                  uint32_t *contour,
                  uint3 contourSize);

void connectParticles(int32_t *adjTriangle,
                      float3 *sortedPos,
                      uint32_t *gridParticleIndex,
                      uint32_t *cellStart,
                      uint32_t *cellEnd,
                      uint32_t numParticles);

void connectClusters(uint32_t *candidatesInds,
                     int32_t *adjTriangle,
                     uint32_t *clusterEdges,
                     bool *clustersMerged,
                     uint32_t numClusters,
                     float minSearchRadius,
                     float maxSearchRadius,
                     float3 *centroids,
                     int32_t *clusterInds,
                     float3 *sortedPos,
                     float3 *oldPos,
                     uint32_t *gridParticleIndex,
                     uint32_t *cellStart,
                     uint32_t *cellEnd);

void calcDegrees(uint32_t *edgesCount,
                 uint32_t *degrees,
                 int32_t *adjTriangle,
                 uint32_t numParticles);

void markIsolatedVertices(char *isolatedVertices,
                          uint32_t *degrees,
                          uint32_t numParticles);

void scanDegreesTh(uint32_t *degrees, uint32_t *scannedDegrees, uint32_t size);

void createAdjList(uint32_t *adjacencyList,
                   int32_t *adjTriangle,
                   uint32_t *edgesOffset,
                   uint32_t *edgesSize,
                   uint32_t numParticles);

void readAdjList(uint32_t *adjacencyList,
                 uint32_t *edgesOffset,
                 uint32_t *edgesSize,
                 uint32_t numParticles);

void nextLayer(float *edgesLengths,
               float *distance,
               uint32_t *verticesDistance,
               bool *frontier,
               int32_t *clusterInds,
               float3 *pos,
               uint32_t *adjacencyList,
               uint32_t *edgesOffset,
               uint32_t *edgesSize,
               uint32_t *currentQueue,
               int32_t level,
               uint32_t currentClusterInd,
               uint32_t queueSize);

void countDegrees(uint32_t *degrees,
                  bool *frontier,
                  uint32_t *adjacencyList,
                  uint32_t *edgesOffset,
                  uint32_t *edgesSize,
                  uint32_t *currentQueue,
                  uint32_t queueSize);

void scanDegrees(uint32_t *degrees, uint32_t *incrDegrees, uint32_t queueSize);

void assignVerticesNextQueue(uint32_t *nextQueue,
                             uint32_t *adjacencyList,
                             uint32_t *edgesOffset,
                             uint32_t *edgesSize,
                             uint32_t *currentQueue,
                             uint32_t *degrees,
                             uint32_t *incrDegrees,
                             bool *frontier,
                             uint32_t queueSize);

void calcClusterCentroid(Cluster *cluster,
                         float3 *oldPos,
                         int32_t *clusterInds,
                         uint32_t currentClusterInd,
                         uint32_t numParticles);

void calcClusterEdgeLengths(Cluster *cluster,
                            float *edgesLengths,
                            uint32_t edgesCount,
                            int32_t *clusterInds,
                            uint32_t currentClusterInd,
                            uint32_t numParticles);

void calcClusterPathLengths(Cluster *cluster,
                            float *distance,
                            uint32_t *verticesDistance,
                            int32_t *clusterInds,
                            uint32_t currentClusterInd,
                            uint32_t numParticles);

void calcClusterBranchingsCount(Cluster *cluster,
                                uint32_t *degrees,
                                int32_t *clusterInds,
                                uint32_t currentClusterInd,
                                uint32_t numParticles);

}

#endif // __KERNEL_WRAPPERS_H__