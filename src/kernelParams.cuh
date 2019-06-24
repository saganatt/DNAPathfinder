#ifndef __KERNEL_PARAMS_H__
#define __KERNEL_PARAMS_H__

#include "vector_types.h"

// constants for kernels
struct KernelParams {
    uint32_t numParticles;
    float searchRadius;
    float clustersSearchRadius;
    float3 voxelSize;

    uint3 gridSize;
    uint32_t numCells;
    float3 worldOrigin;
    float3 cellSize;
};

#endif // __KERNEL_PARAMS_H__
