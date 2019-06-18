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

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"

#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include <helper_functions.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/device_vector.h"
#include "thrust/sort.h"
#include "thrust/scan.h"

#include "particles_kernel_impl.cuh"
#include "BFS.cuh"

extern "C"
{

    void cudaInit(int argc, char **argv)
    {
        int devID;

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        devID = findCudaDevice(argc, (const char **)argv);

        if (devID < 0)
        {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }

    void allocateArray(void **devPtr, size_t size)
    {
        checkCudaErrors(cudaMalloc(devPtr, size));
    }

    void freeArray(void *devPtr)
    {
        checkCudaErrors(cudaFree(devPtr));
    }

    void threadSync()
    {
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void copyArrayToDevice(void *device, const void *host, int offset, int size)
    {
        checkCudaErrors(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
    }

    void copyArrayFromDevice(void *host, const void *device, int size)
    {
        checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
    }

    void setParameters(SimParams *hostParams)
    {
        // copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
    }

    //Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

    void calcHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  float *pos,
                  int    numParticles)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // execute the kernel
        calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,
                                               gridParticleIndex,
                                               (float3 *) pos,
                                               numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     float *sortedPos,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                     float *oldPos,
                                     uint   numParticles,
                                     uint   numCells)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // set all cells to empty
        checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

#if USE_TEX
        checkCudaErrors(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float3)));
#endif

        uint smemSize = sizeof(uint)*(numThreads+1);
        reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
            cellStart,
            cellEnd,
            (float3 *) sortedPos,
            gridParticleHash,
            gridParticleIndex,
            (float3 *) oldPos,
            numParticles);
        getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

#if USE_TEX
        checkCudaErrors(cudaUnbindTexture(oldPosTex));
#endif
    }


    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
    {
        thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
                            thrust::device_ptr<uint>(dGridParticleHash + numParticles),
                            thrust::device_ptr<uint>(dGridParticleIndex));
    }

    void checkContour(int32_t *adjTriangle,
                      uint32_t adjTriangleSize,
                      float * oldPos,
                      uint32_t *contour,
                      uint3 contourSize,
                      uint numParticles)
    {
    #if USE_TEX
        checkCudaErrors(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float3)));
    #endif

        // thread per pair
        uint numThreads, numBlocks;
        computeGridSize(adjTriangleSize, 64, numBlocks, numThreads);

        checkContourD<<< numBlocks, numThreads >>>(adjTriangle,
                    adjTriangleSize,
                    (float3 *) oldPos,
                    contour,
                    contourSize,
                    numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed: checkContourD");

    #if USE_TEX
        checkCudaErrors(cudaUnbindTexture(oldPosTex));
    #endif
    }

    void connectPairs(int32_t *adjTriangle,
                      float *sortedPos,
                      uint *gridParticleIndex,
                      uint *cellStart,
                      uint *cellEnd,
                      uint numParticles)
    {
    #if USE_TEX
        checkCudaErrors(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float3)));
        checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
        checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));
    #endif

        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

        // execute the kernel
        connectPairsD<<< numBlocks, numThreads >>>(adjTriangle,
                (float3 *) sortedPos,
                gridParticleIndex,
                cellStart,
                cellEnd,
                numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed: connectPairsD");

    #if USE_TEX
        checkCudaErrors(cudaUnbindTexture(oldPosTex));
        checkCudaErrors(cudaUnbindTexture(cellStartTex));
        checkCudaErrors(cudaUnbindTexture(cellEndTex));
    #endif
    }

    void calcDegrees(int32_t *adjTriangle,
                     int32_t *edgesCount,
                     int32_t *degrees,
                     uint   numParticles)
    {
        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

        calcDegreesD<<<numBlocks, numThreads >>>(adjTriangle,
            edgesCount,
            degrees,
            numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed: calcDegreesD");
    }

    void markIsolatedVertices(int32_t *degrees,
                              bool *isolatedVertices,
                              uint   numParticles)
    {
        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

        markIsolatedVerticesD<<<numBlocks, numThreads >>>(degrees,
            isolatedVertices,
            numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed: markIsolatedVerticesD");
    }

    void createAdjList(int32_t *adjacencyList,
                       int32_t *adjTriangle,
                       int32_t *edgesOffset,
                       int32_t *edgesSize,
                       uint numParticles)
    {
        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

        createAdjListD<<<numBlocks, numThreads >>>(adjacencyList,
            adjTriangle,
            edgesOffset,
            edgesSize,
            numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed: createAdjListD");
    }

    void readAdjList(int32_t *adjacencyList,
                     int32_t *edgesOffset,
                     int32_t *edgesSize,
                     uint numParticles)
    {
        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

        readAdjListD<<<numBlocks, numThreads >>>(adjacencyList,
            edgesOffset,
            edgesSize,
            numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed: readAdjListD");
    }

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
                   int32_t currentClusterInd)
    {
#if USE_TEX
        checkCudaErrors(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float3)));
#endif

        // thread per queue member
        uint numThreads, numBlocks;
        computeGridSize(queueSize, 64, numBlocks, numThreads);

        // execute the kernel
        nextLayerD<<< numBlocks, numThreads >>>(level,
            adjacencyList,
            edgesOffset,
            edgesSize,
            distance,
            verticesDistance,
            parent,
            queueSize,
            currentQueue,
            (float3 *) oldPos,
            frontier,
            cluster,
            clusterInds,
            currentClusterInd);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed: nextLayerD");

#if USE_TEX
        checkCudaErrors(cudaUnbindTexture(oldPosTex));
#endif
    }

    void completeClusterStats(int32_t *edgesSize,
                              float *oldPos,
                              int numParticles,
                              bool *frontier,
                              Cluster *cluster)
    {
        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

        // execute the kernel
        completeClusterStatsD<<< numBlocks, numThreads >>>(edgesSize,
            (float3 *) oldPos,
            numParticles,
            frontier,
            cluster);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed: completeClusterStatsD");
    }

    void countDegrees(int32_t *adjacencyList,
                      int32_t *edgesOffset,
                      int32_t *edgesSize,
                      int32_t *parent,
                      int queueSize,
                      int32_t *currentQueue,
                      int32_t *degrees,
                      bool *frontier) {
        // thread per queue member
        uint numThreads, numBlocks;
        computeGridSize(queueSize, 64, numBlocks, numThreads);

        // execute the kernel
        countDegreesD<<< numBlocks, numThreads >>>(adjacencyList,
            edgesOffset,
            edgesSize,
            parent,
            queueSize,
            currentQueue,
            degrees,
            frontier);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed: countDegreesD");
    }

    void scanDegreesTh(uint numParticles, int32_t *degrees, int32_t *scannedDegrees) {
        thrust::exclusive_scan(thrust::device_ptr<int32_t>(degrees),
                               thrust::device_ptr<int32_t>(degrees + numParticles),
                               thrust::device_ptr<int32_t>(scannedDegrees));
    }

    void scanDegrees(int queueSize, int32_t *degrees, int32_t *incrDegrees, int32_t *scannedDegrees) {
        // thread per queue member
        uint numThreads, numBlocks;
        computeGridSize(queueSize, 64, numBlocks, numThreads);

        scanDegreesD<<< numBlocks, numThreads >>>(queueSize,
            degrees,
            incrDegrees,
            scannedDegrees);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed: scanDegreesD");
        cudaDeviceSynchronize();

        //count prefix sums on CPU for ends of blocks exclusive
        //already written previous block sum
        incrDegrees[0] = 0;
        for (int i = 64; i < queueSize + 64; i += 64) {
            incrDegrees[i / 64] += incrDegrees[i / 64 - 1];
        }
    }

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
                                 bool *frontier) {
        // thread per queue member
        uint numThreads, numBlocks;
        computeGridSize(queueSize, 64, numBlocks, numThreads);

        // execute the kernel
        assignVerticesNextQueueD<<< numBlocks, numThreads >>>(adjacencyList,
            edgesOffset,
            edgesSize,
            parent,
            queueSize,
            currentQueue,
            nextQueue,
            degrees,
            incrDegrees,
            nextQueueSize,
            frontier);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed: assignVerticesNextQueueD");
    }



}   // extern "C"
