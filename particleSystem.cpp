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
#include <helper_math.h>

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
        float searchRadius, uint32_t *contour, uint3 contourSize, float3 voxelSize) :
    m_bInitialized(false),
    m_numParticles(numParticles),
    m_hAdjTriangle(0),
    m_adjTriangleSize((numParticles * (numParticles - 1)) / 2),
    m_dPos(0),
    m_hContour(contour),
    m_dContour(0),
    m_contourSize(contourSize),
    m_hAdjacencyList(0),
    m_hEdgesOffset(0),
    m_hEdgesSize(0),
    m_hDegrees(0),
    m_hIncrDegrees(0),
    m_hIsolatedVertices(0),
    m_hClusters(),
    m_hClusterInds(0),
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

    m_params.particleRadius = 1.0f; // corresponds to 1A
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

    m_hEdgesCount = 0;

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

    m_hDegrees = new int32_t[m_numParticles];
    memset(m_hDegrees, 0, m_numParticles*sizeof(int32_t));

    m_hIsolatedVertices = new bool[m_numParticles];
    memset(m_hIsolatedVertices, 0, m_numParticles*sizeof(bool));

    m_hClusterInds = new int32_t[m_numParticles];
    memset(m_hClusterInds, -1, m_numParticles*sizeof(int32_t));

    // allocate GPU data
    allocateArray((void **)&m_dPos, 3 * sizeof(float) * m_numParticles);
    allocateArray((void **)&m_dSortedPos, 3 * sizeof(float) * m_numParticles);

    allocateArray((void **)&m_dAdjTriangle, sizeof(int32_t) * m_adjTriangleSize);
    allocateArray((void **)&m_dEdgesCount, sizeof(int32_t));

    allocateArray((void**)&m_dContour, sizeof(uint32_t) * m_contourSize.x * m_contourSize.y * m_contourSize.z);

    allocateArray((void **)&m_dGridParticleHash, m_numParticles*sizeof(uint));
    allocateArray((void **)&m_dGridParticleIndex, m_numParticles*sizeof(uint));

    allocateArray((void **)&m_dCellStart, m_numGridCells*sizeof(uint));
    allocateArray((void **)&m_dCellEnd, m_numGridCells*sizeof(uint));

    allocateArray((void **)&m_dDegrees, sizeof(int32_t) * m_numParticles);

    allocateArray((void **)&m_dEdgesOffset, sizeof(int32_t) * m_numParticles);
    allocateArray((void **)&m_dEdgesSize, sizeof(int32_t) * m_numParticles);

    checkCudaErrors(cudaMallocHost((void**)&m_hIncrDegrees, sizeof(int32_t) * m_numParticles));

    allocateArray((void **)&m_dIsolatedVertices, sizeof(bool) * m_numParticles);

    allocateArray((void **)&m_dClusterInds, sizeof(int32_t) * m_numParticles);
    cudaMemset(m_dClusterInds, -1, m_numParticles*sizeof(int32_t));

    sdkCreateTimer(&m_timer);

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

    delete [] m_hDegrees;
    delete [] m_hIsolatedVertices;

    delete [] m_hClusterInds;

    freeArray(m_dPos);
    freeArray(m_dSortedPos);

    freeArray(m_dAdjTriangle);
    freeArray(m_dEdgesCount);

    freeArray(m_dContour);

    freeArray(m_dGridParticleHash);
    freeArray(m_dGridParticleIndex);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);

    freeArray(m_dAdjacencyList);
    freeArray(m_dEdgesOffset);
    freeArray(m_dEdgesSize);
    freeArray(m_dDegrees);
    checkCudaErrors(cudaFreeHost(m_hIncrDegrees));

    freeArray(m_dIsolatedVertices);

    freeArray(m_dClusterInds);
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
    checkContour(
        m_dAdjTriangle,
        m_adjTriangleSize,
        m_dPos,
        m_dContour,
        m_contourSize,
        m_numParticles);

    //printf("Matching pairs...\n");
    // find pairs of closest points and connect
    connectPairs(
        m_dAdjTriangle,
        m_dSortedPos,
        m_dGridParticleIndex,
        m_dCellStart,
        m_dCellEnd,
        m_numParticles,
        m_numGridCells);

    //printf("Calculating degrees...\n");
    calcDegrees(
        m_dAdjTriangle,
        m_dEdgesCount,
        m_dEdgesSize,
        m_numParticles);

    markIsolatedVertices(
        m_dEdgesSize,
        m_dIsolatedVertices,
        m_numParticles);

    //printf("Converting to adjacency list...\n");
    _convertToAdjList();

    //readAdjList(m_dAdjacencyList,
    //            m_dEdgesOffset,
    //            m_dEdgesSize,
    //            m_numParticles);

    //printf("Doing a BFS...\n");
    _scanBfs();
}

void
ParticleSystem::_convertToAdjList() {
//    getEdgesSize();
//    printf("Vertices degrees:\n");
//    for(int i = 0; i < m_numParticles; i++) {
//        printf("%d: %d\n", i, m_hEdgesSize[i]);
//    }
//    printf("\n\n");
//    dumpAdjTriangle();

    copyArrayFromDevice(&m_hEdgesCount, m_dEdgesCount, sizeof(int32_t));
    m_hEdgesCount /= 2;

    allocateArray((void **)&m_dAdjacencyList, sizeof(int32_t) * 2 * m_hEdgesCount);

    // EdgesSize and EdgesOffset to store permanent values for the whole graph
    scanDegreesTh(m_numParticles, m_dEdgesSize, m_dEdgesOffset);
    createAdjList(m_dAdjacencyList, m_dAdjTriangle, m_dEdgesOffset, m_dEdgesSize, m_numParticles, m_hIncrDegrees);
}

void
ParticleSystem::_scanBfs() {
    //printf("Scan BFS. Initializing variables...\n");

    int32_t* d_currentQueue;
    int32_t* d_nextQueue;
    std::vector<float> h_Distance = std::vector<float>(m_numParticles, std::numeric_limits<float>::max());
    std::vector<int32_t> h_VerticesDistance =
        std::vector<int32_t>(m_numParticles, std::numeric_limits<int32_t>::max());
    std::vector<int32_t> h_Parent = std::vector<int32_t>(m_numParticles, -1);
    float *d_Distance;
    int32_t *d_VerticesDistance;
    int32_t *d_Parent;
    //bool *d_visited;
    bool *d_frontier;

    allocateArray((void**) &d_currentQueue, sizeof(int32_t) * m_numParticles);
    allocateArray((void**) &d_nextQueue, sizeof(int32_t) * m_numParticles);
    allocateArray((void **)&d_Distance, sizeof(int32_t) * m_numParticles);
    allocateArray((void **)&d_VerticesDistance, sizeof(int32_t) * m_numParticles);
    allocateArray((void **)&d_Parent, sizeof(int32_t) * m_numParticles);
    //allocateArray((void **)&d_visited, sizeof(bool) * m_numParticles);
    allocateArray((void **)&d_frontier, sizeof(bool) * m_numParticles);

    Cluster *h_currentCluster = new Cluster[1];
    Cluster *h_initCluster = new Cluster[1];
    h_initCluster->minExtremes = make_float3(std::numeric_limits<float>::max());
    h_initCluster->minExtremesInd = make_int3(-1);
    h_initCluster->maxExtremes = make_float3(std::numeric_limits<float>::min());
    h_initCluster->maxExtremesInd = make_int3(-1);
    h_initCluster->shortestEdge = std::numeric_limits<float>::max();
    h_initCluster->longestPath = 0.0f;
    h_initCluster->longestPathVertices = 0;
    h_initCluster->clusterSize = 1; // First vertex will never be on the frontier (??)
    getEdgesSize();
    h_initCluster->branchingsCount = 0;
    h_initCluster->leavesCount = 0;

    Cluster *d_currentCluster;
    allocateArray((void**) &d_currentCluster, sizeof(Cluster));

    copyArrayFromDevice(m_hIsolatedVertices, m_dIsolatedVertices, m_numParticles * sizeof(bool));

    for(int startVertex = 0; startVertex < m_numParticles; startVertex++) {
        if(m_hIsolatedVertices[startVertex]) {
            //printf("Scan BFS. Continuing because of isolated vertex %d\n", startVertex);
            continue;
        }

        // TODO: Allow repeating to count e.g. longest path?
//        if(startVertex > 0) {
//            copyArrayFromDevice(m_hClusterInds, m_dClusterInds, m_numParticles * sizeof(int32_t));
//            if(m_hClusterInds[startVertex] > -1) {
//                printf("Continuing because of visited vertex %d\n", startVertex);
//                continue;
//            }
//        }

        //printf("Scan BFS. Initializing loop variables...\n");
        //initialize values
        std::fill(h_Distance.begin(), h_Distance.end(), std::numeric_limits<float>::max());
        std::fill(h_VerticesDistance.begin(), h_VerticesDistance.end(), std::numeric_limits<int32_t>::max());
        std::fill(h_Parent.begin(), h_Parent.end(), -1);
        h_Distance[startVertex] = 0.0f;
        h_VerticesDistance[startVertex] = 0;
        h_Parent[startVertex] = 0;

        copyArrayToDevice(d_Distance, h_Distance.data(), 0, m_numParticles * sizeof(float));
        copyArrayToDevice(d_VerticesDistance, h_VerticesDistance.data(), 0, m_numParticles * sizeof(int32_t));
        copyArrayToDevice(d_Parent, h_Parent.data(), 0, m_numParticles * sizeof(int32_t));

        cudaMemset(m_dDegrees, 0, m_numParticles*sizeof(int32_t));
        cudaMemset(m_hIncrDegrees, 0, m_numParticles*sizeof(int32_t));

        cudaMemset(d_currentQueue, 0, m_numParticles*sizeof(int32_t));
        cudaMemset(d_nextQueue, 0, m_numParticles*sizeof(int32_t));
        int32_t firstElementQueue = startVertex;
        copyArrayToDevice(d_currentQueue, &firstElementQueue, 0, sizeof(int32_t));

        int32_t clusterInd = m_hClusters.size();
        copyArrayFromDevice(m_hClusterInds, m_dClusterInds, m_numParticles * sizeof(int32_t));
        if(m_hClusterInds[startVertex] > -1) {
            clusterInd = m_hClusterInds[startVertex];
            copyArrayToDevice(d_currentCluster, &(m_hClusters[clusterInd]), 0, sizeof(Cluster));
        }
        else {
            // Start vertex will never be on the frontier - checking it here
            h_initCluster->branchingsCount = m_hEdgesSize[startVertex] > 2 ? 1 : 0;
            h_initCluster->leavesCount = m_hEdgesSize[startVertex] == 1 ? 1 : 0;
            copyArrayToDevice(d_currentCluster, h_initCluster, 0, sizeof(Cluster));
        }

        //cudaMemset(d_visited, false, m_numParticles*sizeof(bool));
        //bool startVisited = true;
        //copyArrayToDevice(d_visited, &startVisited, startVertex, sizeof(bool));

        int queueSize = 1;
        int nextQueueSize = 0;
        int32_t level = 1;
        //printf("Scan BFS. Starting a BFS from vertex %d...\n", startVertex);
        while (queueSize) {
            cudaMemset(d_frontier, false, m_numParticles*sizeof(bool));
            // next layer phase
            nextLayer(level, m_dAdjacencyList, m_dEdgesOffset, m_dEdgesSize, d_Distance, d_VerticesDistance,
                      d_Parent, queueSize, d_currentQueue, m_dPos, d_frontier,
                      d_currentCluster, m_dClusterInds, clusterInd);
            if(clusterInd == m_hClusters.size()) { // only for new cluster
                completeClusterStats(m_dEdgesSize, m_numParticles, d_frontier, d_currentCluster);
            }
            // counting degrees phase
            countDegrees(m_dAdjacencyList, m_dEdgesOffset, m_dEdgesSize, d_Parent, queueSize, d_currentQueue,
                         m_dDegrees, d_frontier);
            // doing scan on degrees
            //scanDegreesTh(queueSize, m_dDegrees, m_dDegrees);
            scanDegrees(queueSize, m_dDegrees, m_hIncrDegrees, m_dDegrees);
            nextQueueSize = m_hIncrDegrees[(queueSize - 1) / 64 + 1];
            // assigning vertices to nextQueue
            assignVerticesNextQueue(m_dAdjacencyList, m_dEdgesOffset, m_dEdgesSize, d_Parent, queueSize,
                                    d_currentQueue, d_nextQueue, m_dDegrees, m_hIncrDegrees, nextQueueSize,
                                    d_frontier);

            level++;
            queueSize = nextQueueSize;
            std::swap(d_currentQueue, d_nextQueue);
        }

        //printf("Scan BFS. Retrieving cluster stats...\n");
        if(clusterInd < m_hClusters.size()) {
            copyArrayFromDevice(&(m_hClusters[clusterInd]), d_currentCluster, sizeof(Cluster));
        }
        else {
            copyArrayFromDevice(h_currentCluster, d_currentCluster, sizeof(Cluster));
            m_hClusters.emplace_back(*h_currentCluster);
        }
    }

    //printf("Scan BFS. Cleaning...\n");
    freeArray(d_currentQueue);
    freeArray(d_nextQueue);

    freeArray(d_currentCluster);
    delete [] h_currentCluster;
    delete [] h_initCluster;

//    copyArrayFromDevice(h_Distance.data(), d_Distance, m_numParticles * sizeof(int32_t));
//    copyArrayFromDevice(h_Parent.data(), d_Parent, m_numParticles * sizeof(int32_t));
    freeArray(d_Distance);
    freeArray(d_VerticesDistance);
    freeArray(d_Parent);

    //freeArray(d_visited);
    freeArray(d_frontier);

//    printf("Calculated distances:\n");
//    for (int i = 0; i < m_numParticles; i++) {
//        printf("(%d, %d) %f\n", h_Parent[i], i, h_Distance[i]);
//    }

    //dumpClusters();
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
ParticleSystem::dumpAdjTriangle()
{
    int32_t edgesCount = 0;
    getAdjTriangle();
    printf("Adjacency triangle pairs: \n");
    for(int i = 0; i < m_numParticles; i++) {
        for(int j = 0; j < i; j++) {
            int32_t entry = getAdjTriangleEntry(j, i);
            if(entry > 0) {
                printf("(%d, %d)\n", i, j);
                edgesCount++;
            }
        }
        for(int j = i + 1; j < m_numParticles; j++) {
            int32_t entry = getAdjTriangleEntry(i, j);
            if(entry > 0) {
                printf("(%d, %d)\n", i, j);
                edgesCount++;
            }
        }
    }
    edgesCount /= 2;
    printf("\nTotal edges count: %d\n\n", edgesCount);
}

void
ParticleSystem::dumpAdjList()
{
    getAdjList();
    getEdgesOffset();
    getEdgesSize();

    printf("Adjacency list pairs: \n");
    for(int i = 0; i < m_numParticles; i++)
    {
        for(int j = m_hEdgesOffset[i]; j < m_hEdgesOffset[i] + m_hEdgesSize[i]; j++)
        {
            printf("(%d, %d)\n", i, m_hAdjacencyList[j]);
        }
    }
    printf("\n\n");
    printf("Total edges count: %d\n", m_hEdgesCount);
}

void
ParticleSystem::dumpClusters()
{
    printf("Detected clusters: %lu\n", m_hClusters.size());
    for(int i = 0; i < m_hClusters.size(); i++) {
        printf("%d size: %d shortest edge: %f longest edge: %f, longest path: %f longest path in vertices: %d\n"
               "number of branchings: %d number of leaves: %d\n"
               "min: (%f, %f, %f) (%d, %d, %d) max: (%f, %f, %f) (%d, %d, %d)\n",
               i, m_hClusters[i].clusterSize, m_hClusters[i].shortestEdge, m_hClusters[i].longestEdge,
               m_hClusters[i].longestPath, m_hClusters[i].longestPathVertices,
               m_hClusters[i].branchingsCount, m_hClusters[i].leavesCount,
               m_hClusters[i].minExtremes.x, m_hClusters[i].minExtremes.y, m_hClusters[i].minExtremes.z,
               m_hClusters[i].minExtremesInd.x, m_hClusters[i].minExtremesInd.y, m_hClusters[i].minExtremesInd.z,
               m_hClusters[i].maxExtremes.x, m_hClusters[i].maxExtremes.y, m_hClusters[i].maxExtremes.z,
               m_hClusters[i].maxExtremesInd.x, m_hClusters[i].maxExtremesInd.y, m_hClusters[i].maxExtremesInd.z);
    }
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

uint32_t *
ParticleSystem::getContour()
{
    assert(m_bInitialized);

    copyArrayFromDevice(m_hContour, m_dContour, m_contourSize.x * m_contourSize.y * m_contourSize.z * sizeof(uint32_t));
    return m_hContour;
}

void
ParticleSystem::setContour(const uint32_t *p_contour, int start, int count)
{
    assert(m_bInitialized);

    copyArrayToDevice(m_dContour, p_contour, start*sizeof(uint32_t), count*sizeof(uint32_t));
}

int32_t *
ParticleSystem::getAdjTriangle()
{
    assert(m_bInitialized);

    copyArrayFromDevice(m_hAdjTriangle, m_dAdjTriangle, m_adjTriangleSize*sizeof(int32_t));
    return m_hAdjTriangle;
}

int32_t
ParticleSystem::getAdjTriangleEntry(uint32_t row, uint32_t column)
{
    if(row > column) {
        uint32_t tmp = row;
        row = column;
        column = tmp;
    }
    return m_hAdjTriangle[(row * (2 * m_numParticles - 1 - row)) / 2 + column - row - 1];
};

void
ParticleSystem::setAdjTriangleEntry(uint32_t row, uint32_t column, int32_t value)
{
    if(row > column) {
        uint32_t tmp = row;
        row = column;
        column = tmp;
    }
    m_hAdjTriangle[(row * (2 * m_numParticles - 1 - row)) / 2 + column - row - 1] = value;
};

int32_t *
ParticleSystem::getAdjList()
{
    assert(m_bInitialized);
    assert(m_hEdgesCount > 0);

    if(m_hAdjacencyList == nullptr) {
        m_hAdjacencyList = new int32_t[2 * m_hEdgesCount];
    }
    memset(m_hAdjacencyList, 0, m_hEdgesCount*2*sizeof(int32_t));

    copyArrayFromDevice(m_hAdjacencyList, m_dAdjacencyList, m_hEdgesCount*2*sizeof(int32_t));
    return m_hAdjacencyList;
}

int32_t *
ParticleSystem::getEdgesOffset()
{
    assert(m_bInitialized);

    if(m_hEdgesOffset == nullptr) {
        m_hEdgesOffset = new int32_t[m_numParticles];
    }
    memset(m_hEdgesOffset, 0, m_numParticles*sizeof(int32_t));

    copyArrayFromDevice(m_hEdgesOffset, m_dEdgesOffset, m_numParticles*sizeof(int32_t));
    return m_hEdgesOffset;
}

int32_t *
ParticleSystem::getEdgesSize()
{
    assert(m_bInitialized);

    if(m_hEdgesSize == nullptr) {
        m_hEdgesSize = new int32_t[m_numParticles];
    }
    memset(m_hEdgesSize, 0, m_numParticles*sizeof(int32_t));

    copyArrayFromDevice(m_hEdgesSize, m_dEdgesSize, m_numParticles*sizeof(int32_t));
    return m_hEdgesSize;
}

std::vector<Cluster>
ParticleSystem::getClusters()
{
    assert(m_bInitialized);

    return m_hClusters;
}

// Assumes preprocessed adj triangle (max 1 marked column per row <==> directed path)
int32_t *
ParticleSystem::getPairsInd()
{
    getAdjTriangle();
    int32_t *pairsInd = new int32_t[m_numParticles];
    memset(pairsInd, 0, m_numParticles*sizeof(int32_t));

    for(int i = 0; i < m_numParticles; i++) {
        for(int j = i + 1; j < m_numParticles; j++) {
            if(getAdjTriangleEntry(i, j) == 1) {
                pairsInd[i] = j;
                j = m_numParticles; // continue;
            }
        }
    }

    return pairsInd;
}