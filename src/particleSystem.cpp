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
#include "kernelWrappers.cuh"
#include "kernelParams.cuh"

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

#ifdef DEBUG
#if (DEBUG == 1)
#define DEBUG_LEVEL 1
#elif (DEBUG == 2)
#define DEBUG_LEVEL 2
#endif
#else
#define DEBUG_LEVEL 0
#endif

ParticleSystem::ParticleSystem(uint32_t numParticles, uint3 gridSize, const std::vector<float3> &particles,
                               const std::vector<uint32_t> &p_indices, float searchRadius, float clustersSearchRadius,
                               std::vector<uint32_t> &contour, uint3 contourSize, float3 voxelSize) :
        m_bInitialized(false),
        m_hPos(numParticles),
        m_hContour(contour),
        m_contourSize(contourSize),
        m_adjTriangleSize((numParticles * (numParticles - 1)) / 2),
        m_hAdjTriangle(m_adjTriangleSize, 0),
        m_hEdgesCount(nullptr),
        m_hIncrDegrees(nullptr),
        m_hIsolatedVertices(numParticles),
        m_hClusters(0),
        m_hClusterInds(numParticles, -1),
        m_hClusterCentroids(0),
        m_hClustersMerged(nullptr),
        m_timer(nullptr) {
    // Set kernel parameters
    m_params.gridSize = gridSize;
    m_params.numCells = gridSize.x * gridSize.y * gridSize.z;
    m_params.worldOrigin = make_float3(0.0f);
    m_params.cellSize = make_float3(2.0f); // cell size equal to particle diameter (2 A)
    m_params.numParticles = numParticles;
    m_params.searchRadius = searchRadius;
    m_params.clustersSearchRadius = clustersSearchRadius;
    m_params.voxelSize = voxelSize;

    for (int32_t i = 0; i < numParticles; i++) {
        m_hPos[(p_indices[i] - 1)].x = particles[i].x;
        m_hPos[(p_indices[i] - 1)].y = particles[i].y;
        m_hPos[(p_indices[i] - 1)].z = particles[i].z;
    }

    _initialize();
}

ParticleSystem::~ParticleSystem() {
    _finalize();
    m_params.numParticles = 0;
}

void
ParticleSystem::_initialize() {
    assert(!m_bInitialized);

    allocateArray((void **)&m_dPos, sizeof(float3) * m_params.numParticles);
    copyArrayToDevice(m_dPos, m_hPos.data(), m_params.numParticles * sizeof(float3));
    allocateArray((void **)&m_dSortedPos, sizeof(float3) * m_params.numParticles);
    setArray(m_dSortedPos, 0, sizeof(float3) * m_params.numParticles);
    allocateArray((void **)&m_dContour, sizeof(uint32_t) * m_contourSize.x * m_contourSize.y * m_contourSize.z);
    copyArrayToDevice(m_dContour, m_hContour.data(),
                      m_contourSize.x * m_contourSize.y * m_contourSize.z * sizeof(uint32_t));

    allocateArray((void **)&m_dAdjTriangle, sizeof(int32_t) * m_adjTriangleSize);
    setArray(m_dAdjTriangle, 0, m_adjTriangleSize * sizeof(int32_t));

    allocateArray((void **)&m_dEdgesOffset, sizeof(uint32_t) * m_params.numParticles);
    setArray(m_dEdgesOffset, 0, sizeof(uint32_t) * m_params.numParticles);
    allocateArray((void **)&m_dEdgesSize, sizeof(uint32_t) * m_params.numParticles);
    setArray(m_dEdgesSize, 0, sizeof(uint32_t) * m_params.numParticles);
    allocateArray((void **)&m_dIsolatedVertices, sizeof(char) * m_params.numParticles);
    setArray(m_dIsolatedVertices, 0, sizeof(char) * m_params.numParticles);

    allocateArray((void **)&m_dClusterInds, sizeof(int32_t) * m_params.numParticles);
    setArray(m_dClusterInds, -1, m_params.numParticles * sizeof(int32_t));

    allocateArray((void **)&m_dGridParticleHash, m_params.numParticles * sizeof(uint32_t));
    setArray(m_dGridParticleHash, 0, m_params.numParticles * sizeof(uint32_t));
    allocateArray((void **)&m_dGridParticleIndex, m_params.numParticles * sizeof(uint32_t));
    setArray(m_dGridParticleIndex, 0, m_params.numParticles * sizeof(uint32_t));

    allocateArray((void **)&m_dCellStart, m_params.numCells * sizeof(uint32_t));
    setArray(m_dCellStart, 0xffffffff, m_params.numCells * sizeof(uint32_t));
    allocateArray((void **)&m_dCellEnd, m_params.numCells * sizeof(uint32_t));
    setArray(m_dCellEnd, 0xffffffff, m_params.numCells * sizeof(uint32_t));

    checkCudaErrors(cudaMallocHost((void **)&m_hIncrDegrees, sizeof(uint32_t) * m_params.numParticles));
    checkCudaErrors(cudaMallocHost((void **)&m_hClustersMerged, sizeof(bool)));
    checkCudaErrors(cudaMallocHost((void **)&m_hEdgesCount, sizeof(uint32_t)));

    setParameters(&m_params);
    sdkCreateTimer(&m_timer);

    m_bInitialized = true;
}

void
ParticleSystem::_finalize() {
    assert(m_bInitialized);

    freeArray(m_dPos);
    freeArray(m_dSortedPos);
    freeArray(m_dContour);

    freeArray(m_dAdjTriangle);

    freeArray(m_dAdjacencyList);
    freeArray(m_dEdgesOffset);
    freeArray(m_dEdgesSize);
    freeArray(m_dIsolatedVertices);

    freeArray(m_dClusterInds);

    freeArray(m_dClusterCentroids);
    freeArray(m_dClusterCandidatesInds);
    freeArray(m_dClusterEdges);

    freeArray(m_dGridParticleHash);
    freeArray(m_dGridParticleIndex);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);

    checkCudaErrors(cudaFreeHost(m_hIncrDegrees));
    checkCudaErrors(cudaFreeHost(m_hClustersMerged));
    checkCudaErrors(cudaFreeHost(m_hEdgesCount));

    sdkDeleteTimer(&m_timer);
}

void
ParticleSystem::update() {
    assert(m_bInitialized);

    // Calculate grid hash
#if (DEBUG_LEVEL >= 1)
    printf("Calculating the hash\n");
#endif
    calcHash(
            m_dGridParticleHash,
            m_dGridParticleIndex,
            m_dPos,
            m_params.numParticles);

    // Sort particles based on hash
#if (DEBUG_LEVEL >= 1)
    printf("Sorting particles\n");
#endif
    sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_params.numParticles);

    // Reorder particle arrays into sorted order and
    // find start and end of each cell
#if (DEBUG_LEVEL >= 1)
    printf("Reordering data\n");
#endif
    reorderDataAndFindCellStart(
            m_dCellStart,
            m_dCellEnd,
            m_dSortedPos,
            m_dGridParticleHash,
            m_dGridParticleIndex,
            m_dPos,
            m_params.numParticles);

    // Initialise adj triangle - check which pairs can be connected inside the contour
#if (DEBUG_LEVEL >= 1)
    printf("Checking contour\n");
#endif
    checkContour(
            m_dAdjTriangle,
            m_adjTriangleSize,
            m_dPos,
            m_dContour,
            m_contourSize);

    // Find pairs of closest points and connect
#if (DEBUG_LEVEL >= 1)
    printf("Connecting particles\n");
#endif
    connectParticles(
            m_dAdjTriangle,
            m_dSortedPos,
            m_dGridParticleIndex,
            m_dCellStart,
            m_dCellEnd,
            m_params.numParticles);

    // Find degree of each vertex
#if (DEBUG_LEVEL >= 1)
    printf("Calculating degrees\n");
    threadSync();
    sdkResetTimer(&m_timer);
    sdkStartTimer(&m_timer);
#endif
    calcDegrees(
            m_hEdgesCount,
            m_dEdgesSize,
            m_dAdjTriangle,
            m_params.numParticles);
#if (DEBUG_LEVEL >= 1)
    threadSync();
    sdkStopTimer(&m_timer);
    _printTimerInfo();
#endif

    // Find isolated vertices
#if (DEBUG_LEVEL >= 1)
    printf("Finding isolated vertices\n");
    threadSync();
    sdkResetTimer(&m_timer);
    sdkStartTimer(&m_timer);
#endif
    markIsolatedVertices(
            m_dIsolatedVertices,
            m_dEdgesSize,
            m_params.numParticles);
#if (DEBUG_LEVEL >= 1)
    threadSync();
    sdkStopTimer(&m_timer);
    _printTimerInfo();
#endif

    // Create adjacency list
#if (DEBUG_LEVEL >= 1)
    printf("Converting to adj list\n");
    threadSync();
    sdkResetTimer(&m_timer);
    sdkStartTimer(&m_timer);
#endif
    _convertToAdjList();
#if (DEBUG_LEVEL >= 1)
    threadSync();
    sdkStopTimer(&m_timer);
    _printTimerInfo();
#endif

    // Write adjacency list on stdout
#if (DEBUG_LEVEL == 2)
    printf("Adjacency list, number of particles: %d edges count: %d\n",
                m_params.numParticles, m_hEdgesCount[0]);
    readAdjList(m_dAdjacencyList,
                m_dEdgesOffset,
                m_dEdgesSize,
                m_params.numParticles);
#endif

    // Perform a BFS on the graph
#if (DEBUG_LEVEL >= 1)
    printf("Starting BFS\n");
    threadSync();
    sdkResetTimer(&m_timer);
    sdkStartTimer(&m_timer);
#endif
    _scanBfs();
#if (DEBUG_LEVEL >= 1)
    threadSync();
    sdkStopTimer(&m_timer);
    _printTimerInfo();
#endif

    // The main loop: merge clusters until only one is left
    int32_t i = 0;
    do {
#if (DEBUG_LEVEL >= 1)
        printf("Loop %d: %lu clusters\n", i, m_hClusters.size());
#endif

#if (DEBUG_LEVEL >= 1)
        printf("Connecting clusters\n");
        threadSync();
        sdkResetTimer(&m_timer);
        sdkStartTimer(&m_timer);
#endif
        _connectClusters(i > 0);
#if (DEBUG_LEVEL >= 1)
        threadSync();
        sdkStopTimer(&m_timer);
        _printTimerInfo();
#endif

#if (DEBUG_LEVEL >= 1)
        printf("Saving clusters to files\n");
        threadSync();
        sdkResetTimer(&m_timer);
        sdkStartTimer(&m_timer);
        _saveClustersToFiles();
        threadSync();
        sdkStopTimer(&m_timer);
        _printTimerInfo();
#endif

        m_hEdgesCount[0] = 0;
        setArray(m_dEdgesOffset, 0, sizeof(uint32_t) * m_params.numParticles);
        setArray(m_dEdgesSize, 0, sizeof(uint32_t) * m_params.numParticles);
        setArray(m_dIsolatedVertices, 0, sizeof(char) * m_params.numParticles);

#if (DEBUG_LEVEL >= 1)
        printf("Calculating degrees\n");
        threadSync();
        sdkResetTimer(&m_timer);
        sdkStartTimer(&m_timer);
#endif
        calcDegrees(
                m_hEdgesCount,
                m_dEdgesSize,
                m_dAdjTriangle,
                m_params.numParticles);
#if (DEBUG_LEVEL >= 1)
        threadSync();
        sdkStopTimer(&m_timer);
        _printTimerInfo();
#endif

#if (DEBUG_LEVEL >= 1)
        printf("Finding isolated vertices\n");
        threadSync();
        sdkResetTimer(&m_timer);
        sdkStartTimer(&m_timer);
#endif
        markIsolatedVertices(
                m_dIsolatedVertices,
                m_dEdgesSize,
                m_params.numParticles);
#if (DEBUG_LEVEL >= 1)
        threadSync();
        sdkStopTimer(&m_timer);
        _printTimerInfo();
#endif

#if (DEBUG_LEVEL >= 1)
        printf("Converting to adj list\n");
        threadSync();
        sdkResetTimer(&m_timer);
        sdkStartTimer(&m_timer);
#endif
        freeArray(m_dAdjacencyList);
        _convertToAdjList();
#if (DEBUG_LEVEL >= 1)
        threadSync();
        sdkStopTimer(&m_timer);
        _printTimerInfo();
#endif

#if (DEBUG_LEVEL == 2)
        printf("Adjacency list, number of particles: %d edges count: %d\n",
                m_params.numParticles, m_hEdgesCount[0]);
        readAdjList(m_dAdjacencyList,
                    m_dEdgesOffset,
                    m_dEdgesSize,
                    m_params.numParticles);
#endif

#if (DEBUG_LEVEL >= 1)
        printf("Starting BFS\n");
        threadSync();
        sdkResetTimer(&m_timer);
        sdkStartTimer(&m_timer);
#endif
        _scanBfs();
#if (DEBUG_LEVEL >= 1)
        threadSync();
        sdkStopTimer(&m_timer);
        _printTimerInfo();
#endif

        i++;
    } while(m_hClusters.size() > 1);
}

void
ParticleSystem::_convertToAdjList() {
    threadSync();
    m_hEdgesCount[0] /= 2;

#if (DEBUG_LEVEL >= 1)
    printf("Edges count: %d\n", m_hEdgesCount[0]);
#endif

    allocateArray((void **)&m_dAdjacencyList, sizeof(uint32_t) * 2 * m_hEdgesCount[0]);
    setArray(m_dAdjacencyList, 0, sizeof(uint32_t) * 2 * m_hEdgesCount[0]);

    scanDegreesTh(m_dEdgesSize, m_dEdgesOffset, m_params.numParticles);
    createAdjList(m_dAdjacencyList, m_dAdjTriangle, m_dEdgesOffset, m_dEdgesSize, m_params.numParticles);
}

void
ParticleSystem::_initBfs(uint32_t **d_currentQueue, uint32_t **d_nextQueue,
                         float **d_distance, uint32_t **d_verticesDistance,
                         uint32_t **d_degrees, bool **d_frontier, float **d_edgesLengths,
                         Cluster **d_currentCluster, Cluster *h_initCluster) {
    allocateArray((void **)d_currentQueue, sizeof(uint32_t) * m_params.numParticles);
    allocateArray((void **)d_nextQueue, sizeof(uint32_t) * m_params.numParticles);
    allocateArray((void **)d_distance, sizeof(uint32_t) * m_params.numParticles);
    allocateArray((void **)d_verticesDistance, sizeof(uint32_t) * m_params.numParticles);
    allocateArray((void **)d_degrees, sizeof(uint32_t) * m_params.numParticles);
    allocateArray((void **)d_frontier, sizeof(bool) * m_params.numParticles);
    allocateArray((void **)d_edgesLengths, sizeof(float) * m_hEdgesCount[0] * 2);
    allocateArray((void **)d_currentCluster, sizeof(Cluster));

    m_hClusters.clear();
    setArray(m_dClusterInds, -1, m_params.numParticles * sizeof(int32_t));
    m_hClusterCentroids.clear();

    h_initCluster->shortestEdge = std::numeric_limits<float>::max();
    h_initCluster->longestPath = 0.0f;
    h_initCluster->longestPathVertices = 0;
    h_initCluster->clusterSize = 0;
    h_initCluster->branchingsCount = 0;
    h_initCluster->centroid = make_float3(0.0f);

    copyArrayFromDevice(m_hIsolatedVertices.data(), m_dIsolatedVertices,
                        m_params.numParticles * sizeof(char));
}

void
ParticleSystem::_initLoopBfs(std::vector<float> &h_distance, std::vector<uint32_t> &h_verticesDistance,
                             int32_t startVertex, uint32_t **d_currentQueue, uint32_t **d_nextQueue,
                             float **d_distance, uint32_t **d_verticesDistance, uint32_t **d_degrees,
                             float **d_edgesLengths, Cluster **d_currentCluster, Cluster *h_initCluster,
                             uint32_t &clusterInd) {
    copyArrayFromDevice(m_hClusterInds.data(), m_dClusterInds, m_params.numParticles * sizeof(int32_t));
    threadSync();
    if (m_hClusterInds[startVertex] > -1) {
        clusterInd = m_hClusterInds[startVertex];
        copyArrayToDevice(*d_currentCluster, &(m_hClusters[clusterInd]), sizeof(Cluster));
        //return;
    }
    else {
        copyArrayToDevice(*d_currentCluster, h_initCluster, sizeof(Cluster));
    }

    std::fill(h_distance.begin(), h_distance.end(), std::numeric_limits<float>::max());
    std::fill(h_verticesDistance.begin(), h_verticesDistance.end(), std::numeric_limits<uint32_t>::max());
    h_distance[startVertex] = 0.0f;
    h_verticesDistance[startVertex] = 0;

    copyArrayToDevice(*d_distance, h_distance.data(), m_params.numParticles * sizeof(float));
    copyArrayToDevice(*d_verticesDistance, h_verticesDistance.data(),
                      m_params.numParticles * sizeof(uint32_t));

    setArray(*d_degrees, 0, m_params.numParticles * sizeof(uint32_t));
    setArray(m_hIncrDegrees, 0, m_params.numParticles * sizeof(uint32_t));

    setArray(*d_currentQueue, 0, m_params.numParticles * sizeof(uint32_t));
    setArray(*d_nextQueue, 0, m_params.numParticles * sizeof(uint32_t));
    uint32_t firstElementQueue = startVertex;
    copyArrayToDevice(*d_currentQueue, &firstElementQueue, sizeof(uint32_t));

    setArray(*d_edgesLengths, 0, m_hEdgesCount[0] * 2 * sizeof(float));
}

void
ParticleSystem::_scanBfs() {
    uint32_t *d_currentQueue;
    uint32_t *d_nextQueue;
    float *d_distance;
    uint32_t *d_verticesDistance;
    uint32_t *d_degrees;
    bool *d_frontier;
    float *d_edgesLengths;
    Cluster *d_currentCluster;

    Cluster *h_currentCluster = new Cluster[1];
    Cluster *h_initCluster = new Cluster[1];

    std::vector<float> h_distance =
            std::vector<float>(m_params.numParticles, std::numeric_limits<float>::max());
    std::vector<uint32_t> h_verticesDistance =
            std::vector<uint32_t>(m_params.numParticles, std::numeric_limits<uint32_t>::max());

    _initBfs(&d_currentQueue, &d_nextQueue, &d_distance, &d_verticesDistance,
             &d_degrees, &d_frontier, &d_edgesLengths, &d_currentCluster, h_initCluster);
    threadSync();

    for (int32_t startVertex = 0; startVertex < m_params.numParticles; startVertex++) {
        if (m_hIsolatedVertices[startVertex] == 1) {
            continue;
        }

        uint32_t clusterInd = m_hClusters.size();
        _initLoopBfs(h_distance, h_verticesDistance, startVertex, &d_currentQueue, &d_nextQueue,
                     &d_distance, &d_verticesDistance, &d_degrees, &d_edgesLengths, &d_currentCluster,
                     h_initCluster, clusterInd);
        //if(clusterInd < m_hClusters.size()) {
        //    continue;
        //}

        uint32_t queueSize = 1;
        uint32_t nextQueueSize = 0;
        int32_t level = 1;
        while (queueSize) {
            setArray(d_frontier, false, m_params.numParticles * sizeof(bool));
            threadSync();
            nextLayer(
                    d_edgesLengths,
                    d_distance,
                    d_verticesDistance,
                    d_frontier,
                    m_dClusterInds,
                    m_dPos,
                    m_dAdjacencyList,
                    m_dEdgesOffset,
                    m_dEdgesSize,
                    d_currentQueue,
                    level,
                    clusterInd,
                    queueSize);
            countDegrees(
                    d_degrees,
                    d_frontier,
                    m_dAdjacencyList,
                    m_dEdgesOffset,
                    m_dEdgesSize,
                    d_currentQueue,
                    queueSize);
            scanDegrees(d_degrees, m_hIncrDegrees, queueSize);
            nextQueueSize = m_hIncrDegrees[(queueSize - 1) / 64 + 1];
            assignVerticesNextQueue(
                    d_nextQueue,
                    m_dAdjacencyList,
                    m_dEdgesOffset,
                    m_dEdgesSize,
                    d_currentQueue,
                    d_degrees,
                    m_hIncrDegrees,
                    d_frontier,
                    queueSize);
            threadSync();
            level++;
            queueSize = nextQueueSize;
            std::swap(d_currentQueue, d_nextQueue);
        }

        // Repeated on each BFS
        calcClusterPathLengths(
                d_currentCluster,
                d_distance,
                d_verticesDistance,
                m_dClusterInds,
                clusterInd,
                m_params.numParticles);

        if (clusterInd < m_hClusters.size()) {
            copyArrayFromDevice(&(m_hClusters[clusterInd]), d_currentCluster, sizeof(Cluster));
        }
        else {
            // Only for new cluster
            calcClusterCentroid(
                    d_currentCluster,
                    m_dPos,
                    m_dClusterInds,
                    clusterInd,
                    m_params.numParticles);
            calcClusterEdgeLengths(
                    d_currentCluster,
                    d_edgesLengths,
                    m_hEdgesCount[0],
                    m_dClusterInds,
                    clusterInd,
                    m_params.numParticles);
            calcClusterBranchingsCount(
                    d_currentCluster,
                    m_dEdgesSize,
                    m_dClusterInds,
                    clusterInd,
                    m_params.numParticles);
            copyArrayFromDevice(h_currentCluster, d_currentCluster, sizeof(Cluster));
            threadSync();
            h_currentCluster->centroid.x /= h_currentCluster->clusterSize;
            h_currentCluster->centroid.y /= h_currentCluster->clusterSize;
            h_currentCluster->centroid.z /= h_currentCluster->clusterSize;
            m_hClusterCentroids.emplace_back(h_currentCluster->centroid);
            m_hClusters.emplace_back(*h_currentCluster);
        }
    }

    freeArray(d_currentQueue);
    freeArray(d_nextQueue);

    freeArray(d_currentCluster);
    delete[] h_currentCluster;
    delete[] h_initCluster;

    freeArray(d_distance);
    freeArray(d_verticesDistance);
    freeArray(d_degrees);
    freeArray(d_frontier);
    freeArray(d_edgesLengths);
}

void
ParticleSystem::_connectClusters(bool free) {
    if (free) {
        freeArray(m_dClusterCandidatesInds);
        freeArray(m_dClusterEdges);
        freeArray(m_dClusterCentroids);
    }

    allocateArray((void **)&m_dClusterCandidatesInds, sizeof(uint32_t) * 4 * m_hClusterCentroids.size());
    setArray(m_dClusterCandidatesInds, m_params.numParticles, m_hClusterCentroids.size() * 4 * sizeof(uint32_t));

    allocateArray((void **)&m_dClusterEdges, sizeof(uint32_t) * 4 * m_hClusterCentroids.size());
    setArray(m_dClusterEdges, m_params.numParticles, m_hClusterCentroids.size() * 4 * sizeof(uint32_t));

    allocateArray((void **)&m_dClusterCentroids, sizeof(float) * 3 * m_hClusterCentroids.size());
    copyArrayToDevice(m_dClusterCentroids, (float *)m_hClusterCentroids.data(),
                      sizeof(float) * 3 * m_hClusterCentroids.size());

    m_hClustersMerged[0] = false;
    float minSearchRadius = 0.0f;
    float maxSearchRadius = m_params.clustersSearchRadius;
    while (!m_hClustersMerged[0]) {
#if (DEBUG_LEVEL >= 1)
        printf("Not merged yet, radius: %f-%f, number of clusters: %lu\n",
                minSearchRadius, maxSearchRadius, m_hClusters.size());
#endif
        connectClusters(
                m_dClusterCandidatesInds,
                m_dAdjTriangle,
                m_dClusterEdges,
                m_hClustersMerged,
                m_hClusterCentroids.size(),
                minSearchRadius,
                maxSearchRadius,
                m_dClusterCentroids,
                m_dClusterInds,
                m_dSortedPos,
                m_dPos,
                m_dGridParticleIndex,
                m_dCellStart,
                m_dCellEnd);
        threadSync();
        minSearchRadius = maxSearchRadius;
        maxSearchRadius += m_params.clustersSearchRadius;
    }
}

void
ParticleSystem::_saveClustersToFiles() {
    getClusterInds();
    std::vector<uint32_t> clusterCandidatesInd = getClusterCandidatesInd();
    std::vector<uint32_t> edgesOffset = getEdgesOffset();
    std::vector<uint32_t> edgesSize = getEdgesSize();
    std::vector<uint32_t> adjList = getAdjList();

    ColorGenerator colorGen(m_hClusters.size());
    try {
        writeChimeraScriptFromAdjList(adjList, edgesOffset, edgesSize, m_hClusterInds,
                                      m_params.numParticles, colorGen);
        writeChimeraScriptFromClustersCandidatesList(clusterCandidatesInd, m_hClusters.size(),
                                                     m_hClusterInds, m_params.numParticles, colorGen);
        writeClustersCentroidsToPdb(m_hClusterCentroids);
        writeClustersStatsToCsv(m_hClusters);
    }
    catch (std::runtime_error &error) {
        std::cout << error.what() << std::endl;
    }
}

void 
ParticleSystem::_printTimerInfo() {
    float fAvgSeconds = ((float)1.0e-3 * (float)sdkGetTimerValue(&m_timer));

    std::cout << "DNAPathfinder, Throughput = " 
              << std::setprecision(4) << (1.0e-3 * m_params.numParticles) / fAvgSeconds
              << " KParticles/s, Time = " << std::setprecision(5) << fAvgSeconds
              << " s, Size = " << m_params.numParticles << std::endl << std::endl;
}

void
ParticleSystem::dumpAdjTriangle() {
    int32_t edgesCount = 0;
    getAdjTriangle();
    printf("Adjacency triangle pairs: \n");
    for (int32_t i = 0; i < m_params.numParticles; i++) {
        for (int32_t j = 0; j < i; j++) {
            int32_t entry = getAdjTriangleEntry(j, i);
            if (entry > 0) {
                printf("(%d, %d)\n", i, j);
                edgesCount++;
            }
        }
        for (int32_t j = i + 1; j < m_params.numParticles; j++) {
            int32_t entry = getAdjTriangleEntry(i, j);
            if (entry > 0) {
                printf("(%d, %d)\n", i, j);
                edgesCount++;
            }
        }
    }
    edgesCount /= 2;
    printf("\nTotal edges count: %d\n\n", edgesCount);
}

void
ParticleSystem::dumpAdjList() {
    std::vector<uint32_t> adjacencyList = getAdjList();
    std::vector<uint32_t> edgesOffset = getEdgesOffset();
    std::vector<uint32_t> edgesSize = getEdgesSize();

    printf("Adjacency list pairs: \n");
    for (int32_t i = 0; i < m_params.numParticles; i++) {
        for (uint32_t j = edgesOffset[i]; j < edgesOffset[i] + edgesSize[i]; j++) {
            printf("(%d, %d)\n", i, adjacencyList[j]);
        }
    }
    printf("\n\n");
    printf("Total edges count: %d\n", m_hEdgesCount[0]);
}

void
ParticleSystem::dumpClusters() {
    printf("Detected clusters: %lu\n", m_hClusters.size());
    for (int32_t i = 0; i < m_hClusters.size(); i++) {
        printf("%d size: %d shortest edge: %f longest edge: %f "
               "longest path: %f longest path in vertices: %d\n"
               "number of branchings: %d\n"
               "centroid: (%f %f %f)\n",
               i, m_hClusters[i].clusterSize, m_hClusters[i].shortestEdge, m_hClusters[i].longestEdge,
               m_hClusters[i].longestPath, m_hClusters[i].longestPathVertices,
               m_hClusters[i].branchingsCount,
               m_hClusters[i].centroid.x, m_hClusters[i].centroid.y, m_hClusters[i].centroid.z);
    }
    printf("\n");
}

std::vector<int32_t>
ParticleSystem::getAdjTriangle() {
    assert(m_bInitialized);

    copyArrayFromDevice(m_hAdjTriangle.data(), m_dAdjTriangle, m_adjTriangleSize * sizeof(int32_t));
    return std::vector<int32_t>(m_hAdjTriangle);
}

int32_t
ParticleSystem::getAdjTriangleEntry(uint32_t row, uint32_t column) {
    if (row > column) {
        uint32_t tmp = row;
        row = column;
        column = tmp;
    }
    return m_hAdjTriangle[(row * (2 * m_params.numParticles - 1 - row)) / 2 + column - row - 1];
}

void
ParticleSystem::setAdjTriangleEntry(uint32_t row, uint32_t column, int32_t value) {
    if (row > column) {
        uint32_t tmp = row;
        row = column;
        column = tmp;
    }
    m_hAdjTriangle[(row * (2 * m_params.numParticles - 1 - row)) / 2 + column - row - 1] = value;
}

std::vector<uint32_t>
ParticleSystem::getAdjList() {
    assert(m_bInitialized);
    assert(m_hEdgesCount[0] > 0);

    std::vector<uint32_t> adjacencyList(m_hEdgesCount[0] * 2);

    copyArrayFromDevice(adjacencyList.data(), m_dAdjacencyList, m_hEdgesCount[0] * 2 * sizeof(uint32_t));
    return adjacencyList;
}

std::vector<uint32_t>
ParticleSystem::getEdgesOffset() {
    assert(m_bInitialized);

    std::vector<uint32_t> edgesOffset(m_params.numParticles);
    copyArrayFromDevice(edgesOffset.data(), m_dEdgesOffset, m_params.numParticles * sizeof(uint32_t));
    return edgesOffset;
}

std::vector<uint32_t>
ParticleSystem::getEdgesSize() {
    assert(m_bInitialized);

    std::vector<uint32_t> edgesSize(m_params.numParticles);
    copyArrayFromDevice(edgesSize.data(), m_dEdgesSize, m_params.numParticles * sizeof(uint32_t));
    return edgesSize;
}

std::vector<int32_t>
ParticleSystem::getClusterInds() {
    assert(m_bInitialized);

    copyArrayFromDevice(m_hClusterInds.data(), m_dClusterInds, m_params.numParticles * sizeof(int32_t));
    return std::vector<int32_t>(m_hClusterInds);
}

std::vector<float3>
ParticleSystem::getClusterCentroids() {
    assert(m_bInitialized);

    return std::vector<float3>(m_hClusterCentroids);
}

std::vector<Cluster>
ParticleSystem::getClusters() {
    assert(m_bInitialized);

    return std::vector<Cluster>(m_hClusters);
}

std::vector<uint32_t>
ParticleSystem::getClusterCandidatesInd() {
    assert(m_bInitialized);

    std::vector<uint32_t> clusterCandidatesInds(4 * m_hClusterCentroids.size());
    copyArrayFromDevice(clusterCandidatesInds.data(), m_dClusterCandidatesInds,
                        sizeof(uint32_t) * 4 * m_hClusterCentroids.size());

    return clusterCandidatesInds;
}

std::vector<uint32_t>
ParticleSystem::getClusterEdges() {
    assert(m_bInitialized);

    std::vector<uint32_t> clusterEdges(4 * m_hClusterCentroids.size());
    copyArrayFromDevice(clusterEdges.data(), m_dClusterEdges,
                        sizeof(uint32_t) * 4 * m_hClusterCentroids.size());

    return clusterEdges;
}