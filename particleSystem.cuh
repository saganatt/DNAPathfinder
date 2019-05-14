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
                  float *sortedPos,
                  uint32_t *contour,
                  uint3 contourSize,
                  float3 voxelSize,
                  uint   numParticles);

void connectPairs(int32_t *adjTriangle,
             uint32_t adjTriangleSize,
             float *sortedPos,
             uint  *gridParticleIndex,
             uint  *cellStart,
             uint  *cellEnd,
             uint   numParticles,
             uint   numCells);

}