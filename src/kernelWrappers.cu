#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

#include "particlesKernels.cuh"
#include "BFS.cuh"
#include "clusterStatistics.cuh"

extern "C" {
    // Functions from NVIDIA's particles sample
    void cudaInit(int32_t argc, char **argv) {
        int32_t devID;

        // Use command-line specified CUDA device, otherwise use device with highest Gflops/s
        devID = findCudaDevice(argc, (const char **)argv);

        if (devID < 0) {
            throw std::runtime_error("No CUDA Capable devices found, exiting...");
        }
    }

    void freeArray(void *devPtr) {
        checkCudaErrors(cudaFree(devPtr));
    }

    void allocateArray(void **devPtr, size_t size) {
        checkCudaErrors(cudaMalloc(devPtr, size));
    }

    void setArray(void *devPtr, int32_t value, size_t count) {
        checkCudaErrors(cudaMemset(devPtr, value, count));
    }

    void copyArrayToDevice(void *device, const void *host, int32_t size) {
        checkCudaErrors(cudaMemcpy(device, host, size, cudaMemcpyHostToDevice));
    }

    void copyArrayFromDevice(void *host, const void *device, int32_t size) {
        checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
    }

    void setParameters(KernelParams *hostParams) {
        // Copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(KernelParams)));
    }

    // Round a / b to nearest higher integer value
    uint32_t iDivUp(uint32_t a, uint32_t b) {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // Compute grid and thread block size for a given number of elements
    void computeGridSize(uint32_t n, uint32_t blockSize, uint32_t &numBlocks, uint32_t &numThreads) {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

    void calcHash(uint32_t *gridParticleHash,
                  uint32_t *gridParticleIndex,
                  float3 *pos,
                  uint32_t numParticles) {
        uint32_t numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

        calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,
                gridParticleIndex,
                pos);

        getLastCudaError("Kernel execution failed: calcHashD");
    }

    void reorderDataAndFindCellStart(uint32_t  *cellStart,
                                     uint32_t  *cellEnd,
                                     float3 *sortedPos,
                                     uint32_t  *gridParticleHash,
                                     uint32_t  *gridParticleIndex,
                                     float3 *oldPos,
                                     uint32_t   numParticles) {
        uint32_t numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

        uint32_t smemSize = sizeof(uint32_t) * (numThreads + 1);
        reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
                cellStart,
                cellEnd,
                sortedPos,
                gridParticleHash,
                gridParticleIndex,
                oldPos);

        getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");
    }


    void sortParticles(uint32_t *dGridParticleHash, uint32_t *dGridParticleIndex, uint32_t numParticles) {
        thrust::sort_by_key(thrust::device_ptr<uint32_t>(dGridParticleHash),
                            thrust::device_ptr<uint32_t>(dGridParticleHash + numParticles),
                            thrust::device_ptr<uint32_t>(dGridParticleIndex));
    }

    // Custom functions
    void checkContour(int32_t *adjTriangle,
                      uint32_t adjTriangleSize,
                      float3 * oldPos,
                      uint32_t *contour,
                      uint3 contourSize) {
        // Thread per pair
        uint32_t numThreads, numBlocks;
        computeGridSize(adjTriangleSize, 64, numBlocks, numThreads);

        checkContourD<<< numBlocks, numThreads >>>(adjTriangle,
                adjTriangleSize,
                oldPos,
                contour,
                contourSize);

        getLastCudaError("Kernel execution failed: checkContourD");
    }

    void connectParticles(int32_t *adjTriangle,
                          float3 *sortedPos,
                          uint32_t *gridParticleIndex,
                          uint32_t *cellStart,
                          uint32_t *cellEnd,
                          uint32_t numParticles) {
        // Thread per particle
        uint32_t numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

        connectParticlesD<<< numBlocks, numThreads >>>(adjTriangle,
                sortedPos,
                gridParticleIndex,
                cellStart,
                cellEnd);

        getLastCudaError("Kernel execution failed: connectParticlesD");
    }

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
                         uint32_t *cellEnd) {
        // Thread per particle
        uint32_t numThreads, numBlocks;
        computeGridSize(numClusters, 64, numBlocks, numThreads);

        connectClustersD<<< numBlocks, numThreads >>>(candidatesInds,
                adjTriangle,
                clusterEdges,
                clustersMerged,
                numClusters,
                minSearchRadius,
                maxSearchRadius,
                centroids,
                clusterInds,
                sortedPos,
                oldPos,
                gridParticleIndex,
                cellStart,
                cellEnd);

        getLastCudaError("Kernel execution failed: connectClustersD");
    }

    void calcDegrees(uint32_t *edgesCount,
                     uint32_t *degrees,
                     int32_t *adjTriangle,
                     uint32_t numParticles) {
        // Thread per particle
        uint32_t numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

        calcDegreesD<<< numBlocks, numThreads >>>(edgesCount,
                degrees,
                adjTriangle);

        getLastCudaError("Kernel execution failed: calcDegreesD");
    }

    void markIsolatedVertices(char *isolatedVertices,
                              uint32_t *degrees,
                              uint32_t numParticles) {
        // Thread per particle
        uint32_t numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

        markIsolatedVerticesD<<< numBlocks, numThreads >>>(isolatedVertices, degrees);

        getLastCudaError("Kernel execution failed: markIsolatedVerticesD");
    }

    void scanDegreesTh(uint32_t *degrees, uint32_t *scannedDegrees, uint32_t size) {
        thrust::exclusive_scan(thrust::device_ptr<uint32_t>(degrees),
                               thrust::device_ptr<uint32_t>(degrees + size),
                               thrust::device_ptr<uint32_t>(scannedDegrees));
    }

    void createAdjList(uint32_t *adjacencyList,
                       int32_t *adjTriangle,
                       uint32_t *edgesOffset,
                       uint32_t *edgesSize,
                       uint32_t numParticles) {
        // Thread per particle
        uint32_t numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

        createAdjListD<<< numBlocks, numThreads >>>(adjacencyList,
                adjTriangle,
                edgesOffset,
                edgesSize);

        getLastCudaError("Kernel execution failed: createAdjListD");
    }

    void readAdjList(uint32_t *adjacencyList,
                     uint32_t *edgesOffset,
                     uint32_t *edgesSize,
                     uint32_t numParticles) {
        // Thread per particle
        uint32_t numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

        readAdjListD<<< numBlocks, numThreads >>>(adjacencyList,
                edgesOffset,
                edgesSize);

        getLastCudaError("Kernel execution failed: readAdjListD");
    }

    void nextLayer(float *edgesLengths,
                   float *distance,
                   uint32_t *verticesDistance,
                   char *frontier,
                   int32_t *clusterInds,
                   float3 *pos,
                   uint32_t *adjacencyList,
                   uint32_t *edgesOffset,
                   uint32_t *edgesSize,
                   uint32_t *currentQueue,
                   int32_t level,
                   uint32_t currentClusterInd,
                   uint32_t queueSize) {
        // Thread per queue member
        uint32_t numThreads, numBlocks;
        computeGridSize(queueSize, 64, numBlocks, numThreads);

        nextLayerD<<< numBlocks, numThreads >>>(edgesLengths,
                distance,
                verticesDistance,
                frontier,
                clusterInds,
                pos,
                adjacencyList,
                edgesOffset,
                edgesSize,
                currentQueue,
                level,
                currentClusterInd,
                queueSize);

        getLastCudaError("Kernel execution failed: nextLayerD");
    }

    void countDegrees(uint32_t *degrees,
                      char *frontier,
                      uint32_t *adjacencyList,
                      uint32_t *edgesOffset,
                      uint32_t *edgesSize,
                      uint32_t *currentQueue,
                      uint32_t queueSize) {
        // Thread per queue member
        uint32_t numThreads, numBlocks;
        computeGridSize(queueSize, 64, numBlocks, numThreads);

        countDegreesD<<< numBlocks, numThreads >>>(degrees,
                frontier,
                adjacencyList,
                edgesOffset,
                edgesSize,
                currentQueue,
                queueSize);

        getLastCudaError("Kernel execution failed: countDegreesD");
    }

    void countDegrees2(uint32_t *degrees,
                       char *frontier,
                       uint32_t *adjacencyList,
                       uint32_t *edgesOffset,
                       uint32_t *edgesSize,
                       uint32_t verticesCount) {
        // Thread per vertex
        uint32_t numThreads, numBlocks;
        computeGridSize(verticesCount, 64, numBlocks, numThreads);

        countDegrees2D<<< numBlocks, numThreads >>>(degrees,
                frontier,
                adjacencyList,
                edgesOffset,
                edgesSize,
                verticesCount);

        getLastCudaError("Kernel execution failed: countDegrees2D");
    }

    void scanDegrees(uint32_t *degrees, uint32_t *incrDegrees, uint32_t queueSize) {
        // Thread per queue member
        uint32_t numThreads, numBlocks;
        computeGridSize(queueSize, 64, numBlocks, numThreads);

        scanDegreesD<<< numBlocks, numThreads >>>(degrees,
            incrDegrees,
            queueSize);

        checkCudaErrors(cudaDeviceSynchronize());
        getLastCudaError("Kernel execution failed: scanDegreesD");

        // Count prefix sums on CPU for ends of blocks exclusive
        // Already written previous block sum
        incrDegrees[0] = 0;
        for (int32_t i = 64; i < queueSize + 64; i += 64) {
            incrDegrees[i / 64] += incrDegrees[i / 64 - 1];
        }
    }

    void assignVerticesNextQueue(uint32_t *nextQueue,
                                 uint32_t *adjacencyList,
                                 uint32_t *edgesOffset,
                                 uint32_t *edgesSize,
                                 uint32_t *currentQueue,
                                 uint32_t *degrees,
                                 uint32_t *incrDegrees,
                                 char *frontier,
                                 uint32_t queueSize) {
        // Thread per queue member
        uint32_t numThreads, numBlocks;
        computeGridSize(queueSize, 64, numBlocks, numThreads);

        assignVerticesNextQueueD<<< numBlocks, numThreads >>>(nextQueue,
            adjacencyList,
            edgesOffset,
            edgesSize,
            currentQueue,
            degrees,
            incrDegrees,
            frontier,
            queueSize);

        getLastCudaError("Kernel execution failed: assignVerticesNextQueueD");
    }

    void assignVerticesNextQueue2(uint32_t *nextQueue,
                                  uint32_t *adjacencyList,
                                  uint32_t *edgesOffset,
                                  uint32_t *edgesSize,
                                  uint32_t *degrees,
                                  uint32_t *incrDegrees,
                                  char *frontier,
                                  uint32_t verticesCount) {
        // Thread per vertex
        uint32_t numThreads, numBlocks;
        computeGridSize(verticesCount, 64, numBlocks, numThreads);

        assignVerticesNextQueue2D<<< numBlocks, numThreads >>>(nextQueue,
                adjacencyList,
                edgesOffset,
                edgesSize,
                degrees,
                incrDegrees,
                frontier,
                verticesCount);

        getLastCudaError("Kernel execution failed: assignVerticesNextQueue2D");
    }

    // Functions adapted from NVIDIA's reduce sample

    // Find closest power of 2 not smaller than given value
    uint32_t nextPow2(uint32_t x)
    {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }

    // Check if value is power of 2
    bool isPow2(uint32_t x)
    {
        return ((x & (x - 1)) == 0);
    }

    // Compute grid and thread block size that are powers of 2
    void computeGridSizePow2(uint32_t n, uint32_t blockSize, uint32_t gridSize,
                             uint32_t &numBlocks, uint32_t &numThreads) {
        numThreads = (n < blockSize * 2) ? nextPow2((n + 1) / 2) : blockSize;
        numBlocks = (n + (numThreads * 2 - 1)) / (numThreads * 2);
        numBlocks = min(gridSize, numBlocks);
    }

    void calcClusterCentroid(Cluster *cluster,
                             float3 *oldPos,
                             int32_t *clusterInds,
                             uint32_t currentClusterInd,
                             uint32_t numParticles) {
        // Thread per particle
        uint32_t numThreads, numBlocks;
        computeGridSizePow2(numParticles, 64, 64, numBlocks, numThreads);

        // When there is only one warp per block, we need to allocate two warps
        // worth of shared memory for each array so that we don't index shared memory out of bounds
        uint32_t smemMul = (numThreads <= 32) ? 2 * numThreads : numThreads;
        uint32_t smemSize = smemMul * (sizeof(float3) + sizeof(uint32_t));

        if (isPow2(numParticles)) {
            switch(numThreads) {
                case 1024:
                    calcClusterCentroidD<1024, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 512:
                    calcClusterCentroidD<512, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 256:
                    calcClusterCentroidD<256, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 128:
                    calcClusterCentroidD<128, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 64:
                    calcClusterCentroidD<64, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 32:
                    calcClusterCentroidD<32, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 16:
                    calcClusterCentroidD<16, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 8:
                    calcClusterCentroidD<8, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 4:
                    calcClusterCentroidD<4, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 2:
                    calcClusterCentroidD<2, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 1:
                default:
                    calcClusterCentroidD<1, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
            }
        }
        else {
            switch(numThreads) {
                case 1024:
                    calcClusterCentroidD<1024, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 512:
                    calcClusterCentroidD<512, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 256:
                    calcClusterCentroidD<256, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 128:
                    calcClusterCentroidD<128, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 64:
                    calcClusterCentroidD<64, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 32:
                    calcClusterCentroidD<32, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 16:
                    calcClusterCentroidD<16, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 8:
                    calcClusterCentroidD<8, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 4:
                    calcClusterCentroidD<4, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 2:
                    calcClusterCentroidD<2, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 1:
                default:
                    calcClusterCentroidD<1, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            oldPos,
                            clusterInds,
                            currentClusterInd);
                    break;
            }
        }

        getLastCudaError("Kernel execution failed: calcClusterCentroidD");
    }

    void calcClusterEdgeLengths(Cluster *cluster,
                                float *edgesLengths,
                                uint32_t edgesCount) {
        // Thread per particle
        uint32_t numThreads, numBlocks;
        computeGridSizePow2(edgesCount, 64, 64, numBlocks, numThreads);

        // When there is only one warp per block, we need to allocate two warps
        // worth of shared memory for each array so that we don't index shared memory out of bounds
        uint32_t smemMul = (numThreads <= 32) ? 2 * numThreads : numThreads;
        uint32_t smemSize = smemMul * (sizeof(float3) + sizeof(uint32_t));

        if (isPow2(edgesCount)) {
            switch(numThreads) {
                case 1024:
                    calcClusterEdgeLengthsD<1024, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
                case 512:
                    calcClusterEdgeLengthsD<512, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
                case 256:
                    calcClusterEdgeLengthsD<256, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
                case 128:
                    calcClusterEdgeLengthsD<128, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
                case 64:
                    calcClusterEdgeLengthsD<64, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
                case 32:
                    calcClusterEdgeLengthsD<32, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
                case 16:
                    calcClusterEdgeLengthsD<16, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
                case 8:
                    calcClusterEdgeLengthsD<8, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
                case 4:
                    calcClusterEdgeLengthsD<4, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
                case 2:
                    calcClusterEdgeLengthsD<2, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
                case 1:
                default:
                    calcClusterEdgeLengthsD<1, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
            }
        }
        else {
            switch(numThreads) {
                case 1024:
                    calcClusterEdgeLengthsD<1024, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
                case 512:
                    calcClusterEdgeLengthsD<512, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
                case 256:
                    calcClusterEdgeLengthsD<256, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
                case 128:
                    calcClusterEdgeLengthsD<128, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
                case 64:
                    calcClusterEdgeLengthsD<64, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
                case 32:
                    calcClusterEdgeLengthsD<32, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
                case 16:
                    calcClusterEdgeLengthsD<16, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
                case 8:
                    calcClusterEdgeLengthsD<8, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
                case 4:
                    calcClusterEdgeLengthsD<4, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
                case 2:
                    calcClusterEdgeLengthsD<2, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
                case 1:
                default:
                    calcClusterEdgeLengthsD<1, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            edgesLengths,
                            edgesCount);
                    break;
            }
        }

        getLastCudaError("Kernel execution failed: calcClusterEdgeLengthsD");
    }

    void calcClusterPathLengths(Cluster *cluster,
                                float *distance,
                                uint32_t *verticesDistance,
                                int32_t *clusterInds,
                                uint32_t currentClusterInd,
                                uint32_t numParticles) {
        // Thread per particle
        uint32_t numThreads, numBlocks;
        computeGridSizePow2(numParticles, 64, 64, numBlocks, numThreads);

        // When there is only one warp per block, we need to allocate two warps
        // worth of shared memory for each array so that we don't index shared memory out of bounds
        uint32_t smemMul = (numThreads <= 32) ? 2 * numThreads : numThreads;
        uint32_t smemSize = smemMul * (sizeof(float3) + sizeof(uint32_t));

        if (isPow2(numParticles)) {
            switch(numThreads) {
                case 1024:
                    calcClusterPathLengthsD<1024, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 512:
                    calcClusterPathLengthsD<512, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 256:
                    calcClusterPathLengthsD<256, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 128:
                    calcClusterPathLengthsD<128, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 64:
                    calcClusterPathLengthsD<64, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 32:
                    calcClusterPathLengthsD<32, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 16:
                    calcClusterPathLengthsD<16, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 8:
                    calcClusterPathLengthsD<8, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 4:
                    calcClusterPathLengthsD<4, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 2:
                    calcClusterPathLengthsD<2, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 1:
                default:
                    calcClusterPathLengthsD<1, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
            }
        }
        else {
            switch(numThreads) {
                case 1024:
                    calcClusterPathLengthsD<1024, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 512:
                    calcClusterPathLengthsD<512, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 256:
                    calcClusterPathLengthsD<256, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 128:
                    calcClusterPathLengthsD<128, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 64:
                    calcClusterPathLengthsD<64, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 32:
                    calcClusterPathLengthsD<32, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 16:
                    calcClusterPathLengthsD<16, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 8:
                    calcClusterPathLengthsD<8, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 4:
                    calcClusterPathLengthsD<4, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 2:
                    calcClusterPathLengthsD<2, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 1:
                default:
                    calcClusterPathLengthsD<1, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            distance,
                            verticesDistance,
                            clusterInds,
                            currentClusterInd);
                    break;
            }
        }

        getLastCudaError("Kernel execution failed: calcClusterPathLengthsD");
    }

    void calcClusterBranchingsCount(Cluster *cluster,
                                    uint32_t *degrees,
                                    int32_t *clusterInds,
                                    uint32_t currentClusterInd,
                                    uint32_t numParticles) {
        // Thread per particle
        uint32_t numThreads, numBlocks;
        computeGridSizePow2(numParticles, 64, 64, numBlocks, numThreads);

        // When there is only one warp per block, we need to allocate two warps
        // worth of shared memory for each array so that we don't index shared memory out of bounds
        uint32_t smemMul = (numThreads <= 32) ? 2 * numThreads : numThreads;
        uint32_t smemSize = smemMul * (sizeof(float3) + sizeof(uint32_t));

        if (isPow2(numParticles)) {
            switch(numThreads) {
                case 1024:
                    calcClusterBranchingsCountD<1024, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 512:
                    calcClusterBranchingsCountD<512, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 256:
                    calcClusterBranchingsCountD<256, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 128:
                    calcClusterBranchingsCountD<128, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 64:
                    calcClusterBranchingsCountD<64, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 32:
                    calcClusterBranchingsCountD<32, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 16:
                    calcClusterBranchingsCountD<16, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 8:
                    calcClusterBranchingsCountD<8, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 4:
                    calcClusterBranchingsCountD<4, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 2:
                    calcClusterBranchingsCountD<2, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 1:
                default:
                    calcClusterBranchingsCountD<1, true> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
            }
        }
        else {
            switch(numThreads) {
                case 1024:
                    calcClusterBranchingsCountD<1024, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 512:
                    calcClusterBranchingsCountD<512, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 256:
                    calcClusterBranchingsCountD<256, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 128:
                    calcClusterBranchingsCountD<128, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 64:
                    calcClusterBranchingsCountD<64, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 32:
                    calcClusterBranchingsCountD<32, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 16:
                    calcClusterBranchingsCountD<16, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 8:
                    calcClusterBranchingsCountD<8, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 4:
                    calcClusterBranchingsCountD<4, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 2:
                    calcClusterBranchingsCountD<2, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
                case 1:
                default:
                    calcClusterBranchingsCountD<1, false> << < numBlocks, numThreads, smemSize >> > (cluster,
                            degrees,
                            clusterInds,
                            currentClusterInd);
                    break;
            }
        }

        getLastCudaError("Kernel execution failed: calcClusterBranchingsCountD");
    }

}   // extern "C"
