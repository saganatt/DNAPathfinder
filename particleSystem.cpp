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

#include "particleSystem.h"
#include "particleSystem.cuh"
#include "particles_kernel.cuh"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize, float *particles, uint32_t *p_indices,
        float searchRadius, uint32_t *contour, uint3 contourSize, float3 voxelSize, float maxCoordinate) :
    m_bInitialized(false),
    m_numParticles(numParticles),
    m_hAdjTriangle(0),
    m_adjTriangleSize((numParticles * (numParticles - 1)) / 2),
    m_dPos(0),
    m_hContour(contour),
    m_dContour(0),
    m_contourSize(contourSize),
    m_gridSize(gridSize),
    m_timer(NULL)
{
    m_numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;
    //    float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

    m_gridSortBits = 18;    // increase this for larger grids

    // set simulation parameters
    m_params.gridSize = m_gridSize;
    m_params.numCells = m_numGridCells;
    m_params.numBodies = m_numParticles;

    m_params.particleRadius = 1.0f / maxCoordinate; // corresponds to 1A
    m_params.searchRadius = searchRadius;

    m_params.voxelSize = voxelSize;

    m_params.worldOrigin = make_float3(0.0f, 0.0f, 0.0f); //(-1.0f, -1.0f, -1.0f);
    //    m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
    float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
    m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

    m_hPos = new float[numParticles * 3];
    for(int i = 0; i < numParticles; i++) {
        m_hPos[3 * (p_indices[i] - 1)] = particles[3 * i];
        m_hPos[3 * (p_indices[i] - 1) + 1] = particles[3 * i + 1];
        m_hPos[3 * (p_indices[i] - 1) + 2] = particles[3 * i + 2];
    }

    _initialize();
}

ParticleSystem::~ParticleSystem()
{
    _finalize();
    m_numParticles = 0;
}

void
ParticleSystem::_initialize()
{
    assert(!m_bInitialized);

    // allocate host storage
    m_hCellStart = new uint[m_numGridCells];
    memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

    m_hCellEnd = new uint[m_numGridCells];
    memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));

    m_hAdjTriangle = new int32_t[m_adjTriangleSize];
    memset(m_hAdjTriangle, 0, m_adjTriangleSize*sizeof(int32_t));

    // allocate GPU data
    allocateArray((void **)&m_dPos, 3 * sizeof(float) * m_numParticles);
    allocateArray((void **)&m_dSortedPos, 3 * sizeof(float) * m_numParticles);

    allocateArray((void **)&m_dAdjTriangle, sizeof(int32_t) * m_adjTriangleSize);

    allocateArray((void**)&m_dContour, sizeof(uint32_t) * m_contourSize.x * m_contourSize.y * m_contourSize.z);

    allocateArray((void **)&m_dGridParticleHash, m_numParticles*sizeof(uint));
    allocateArray((void **)&m_dGridParticleIndex, m_numParticles*sizeof(uint));

    allocateArray((void **)&m_dCellStart, m_numGridCells*sizeof(uint));
    allocateArray((void **)&m_dCellEnd, m_numGridCells*sizeof(uint));

    sdkCreateTimer(&m_timer);
\
    setParameters(&m_params);

    m_bInitialized = true;

    setPositions(m_hPos, 0, m_numParticles);
    setContour(m_hContour, 0, m_contourSize.x * m_contourSize.y * m_contourSize.z);
}

void
ParticleSystem::_finalize()
{
    assert(m_bInitialized);

    delete [] m_hPos;
    delete [] m_hAdjTriangle;
    delete [] m_hContour;
    delete [] m_hCellStart;
    delete [] m_hCellEnd;

    freeArray(m_dPos);
    freeArray(m_dSortedPos);

    freeArray(m_dAdjTriangle);

    freeArray(m_dContour);

    freeArray(m_dGridParticleHash);
    freeArray(m_dGridParticleIndex);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);
}

// step the simulation
void
ParticleSystem::update()
{
    assert(m_bInitialized);

    // update constants
    setParameters(&m_params);

    // calculate grid hash
    calcHash(
        m_dGridParticleHash,
        m_dGridParticleIndex,
        m_dPos,
        m_numParticles);

    // sort particles based on hash
    sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

    // reorder particle arrays into sorted order and
    // find start and end of each cell
    reorderDataAndFindCellStart(
        m_dCellStart,
        m_dCellEnd,
        m_dSortedPos,
        m_dGridParticleHash,
        m_dGridParticleIndex,
        m_dPos,
        m_numParticles,
        m_numGridCells);

    // initialise adj triangle - check which pairs can be connected inside the contour
    checkContour(m_dAdjTriangle,
                 m_adjTriangleSize,
                 m_dPos,
                 m_dContour,
                 m_contourSize,
                 m_params.voxelSize,
                 m_numParticles);

    // find pairs of closest points and connect
    connectPairs(
            m_dAdjTriangle,
            m_adjTriangleSize,
            m_dSortedPos,
            m_dGridParticleIndex,
            m_dCellStart,
            m_dCellEnd,
            m_numParticles,
            m_numGridCells);
}

void
ParticleSystem::dumpGrid()
{
    // dump grid information
    copyArrayFromDevice(m_hCellStart, m_dCellStart, sizeof(uint)*m_numGridCells);
    copyArrayFromDevice(m_hCellEnd, m_dCellEnd, sizeof(uint)*m_numGridCells);
    uint maxCellSize = 0;

    for (uint i=0; i<m_numGridCells; i++)
    {
        if (m_hCellStart[i] != 0xffffffff)
        {
            uint cellSize = m_hCellEnd[i] - m_hCellStart[i];

            //            printf("cell: %d, %d particles\n", i, cellSize);
            if (cellSize > maxCellSize)
            {
                maxCellSize = cellSize;
            }
        }
    }

    printf("maximum particles per cell = %d\n", maxCellSize);
}

void
ParticleSystem::dumpParticles(uint start, uint count)
{
    // debug
    m_hPos = getPositions();

    for (uint i=start; i<start+count; i++)
    {
        printf("%u %.4f %.4f %.4f\n", i, m_hPos[i*3+0], m_hPos[i*3+1], m_hPos[i*3+2]);
    }
}

float *
ParticleSystem::getPositions()
{
    assert(m_bInitialized);

    copyArrayFromDevice(m_hPos, m_dPos, m_numParticles*3*sizeof(float));
    return m_hPos;
}

void
ParticleSystem::setPositions(const float *position, int start, int count)
{
    assert(m_bInitialized);

    copyArrayToDevice(m_dPos, position, start*3*sizeof(float), count*3*sizeof(float));
}

int32_t *
ParticleSystem::getAdjTriangle() {
    assert(m_bInitialized);

    copyArrayFromDevice(m_hAdjTriangle, m_dAdjTriangle, m_adjTriangleSize*sizeof(int32_t));
    return m_hAdjTriangle;
}

int32_t
ParticleSystem::getAdjTriangleEntry(int32_t *adjTriangle, uint32_t row, uint32_t column) {
    return adjTriangle[(row * (2 * m_numParticles - 1 - row)) / 2 + column - row - 1];
};

void
ParticleSystem::setAdjTriangleEntry(int32_t *adjTriangle, uint32_t row, uint32_t column, int32_t value) {
    adjTriangle[(row * (2 * m_numParticles - 1 - row)) / 2 + column - row - 1] = value;
};

// Assumes preprocessed adj triangle (max 1 marked column per row <==> directed path)
int32_t *
ParticleSystem::getPairsInd() {
    int32_t *adjTriangle = getAdjTriangle();
    int32_t *pairsInd = new int32_t[m_numParticles];
    memset(pairsInd, 0, m_numParticles*sizeof(int32_t));

    for(int i = 0; i < m_numParticles; i++) {
        for(int j = i + 1; j < m_numParticles; j++) {
            if(getAdjTriangleEntry(adjTriangle, i, j) == 1) {
                pairsInd[i] = j;
                j = m_numParticles; // continue;
            }
        }
    }

    return pairsInd;
}

uint32_t *
ParticleSystem::getContour() {
    assert(m_bInitialized);

    copyArrayFromDevice(m_hContour, m_dContour, m_contourSize.x * m_contourSize.y * m_contourSize.z * sizeof(uint32_t));
    return m_hContour;
}

void
ParticleSystem::setContour(const uint32_t *p_contour, int start, int count){
    assert(m_bInitialized);

    copyArrayToDevice(m_dContour, p_contour, start*sizeof(uint32_t), count*sizeof(uint32_t));
}
