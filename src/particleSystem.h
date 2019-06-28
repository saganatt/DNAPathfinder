#ifndef __PARTICLE_SYSTEM_H__
#define __PARTICLE_SYSTEM_H__

#include <stdint.h>

#include "kernelParams.cuh"
#include "cluster.h"
#include "filesIO.h"

// Particle system class
class ParticleSystem {
    public:
        ParticleSystem(uint32_t numParticles, uint3 gridSize, const std::vector<float3> &particles,
                       const std::vector<uint32_t> &p_indices, float searchRadius, float clustersSearchRadius,
                       std::vector<uint32_t> &contour, uint3 contourSize, float3 voxelSize, int32_t loops,
                       bool useContour);

        ~ParticleSystem();

        void update();

        std::vector<int32_t> getAdjTriangle();

        int32_t getAdjTriangleEntry(uint32_t row, uint32_t column);

        void setAdjTriangleEntry(uint32_t row, uint32_t column, int32_t value);

        std::vector<uint32_t> getAdjList();

        std::vector<uint32_t> getEdgesOffset();

        std::vector<uint32_t> getEdgesSize();

        std::vector<int32_t> getClusterInds();

        std::vector<float3> getClusterCentroids();

        std::vector<Cluster> getClusters();

        std::vector<uint32_t> getClusterCandidatesInd();

        std::vector<uint32_t> getClusterEdges();

        void dumpAdjTriangle();

        void dumpAdjList();

        void dumpClusters();

    private: // methods
        void _initialize();

        void _finalize();

        void _convertToAdjList();

        void _initBfs(uint32_t **d_currentQueue, uint32_t **d_nextQueue,
                      float **d_distance, uint32_t **d_verticesDistance, char **d_frontier,
                      uint32_t **d_degrees, float **d_edgesLengths,
                      Cluster **d_currentCluster, Cluster *h_initCluster);

        void _initLoopBfs(std::vector<float> &h_distance, std::vector<uint32_t> &h_verticesDistance,
                          int32_t startVertex, uint32_t **d_currentQueue, uint32_t **d_nextQueue,
                          float **d_distance, uint32_t **d_verticesDistance, char **d_frontier,
                          uint32_t **d_degrees, float **edgesLengths,
                          Cluster **d_currentCluster, Cluster *h_initCluster, uint32_t &clusterInd);

        void _scanBfs();

        void _connectClusters(bool free);

        void _saveClustersToFiles();

        void _printTimerInfo(std::string str = "");

    private: // data
        bool m_bInitialized;

        int32_t m_loopsNumber;                      // max number of main loops to execute
                                                    // 0 means infinite (loop until all clusters merged)

        // CPU data
        std::vector<float3> m_hPos;                 // particle positions
        // Contour access at (x, y, z) (starting from 0):
        //    x + y * contourSize.x + z * contourSize.x * contourSize.y;
        // Multiply each coordinate by voxelSize to get world coordinates of given matrix cell
        std::vector<uint32_t> m_hContour;           // values from contour matrix
        uint3 m_contourSize;                        // matrix sizes in each dimension
        bool m_useContour;                          // whether there is contour to be processed

        uint32_t m_adjTriangleSize;                 // size of adjacency triangle
        std::vector<int32_t> m_hAdjTriangle;        // upper half of adjacency matrix without main diagonal
        uint32_t *m_hEdgesCount;                    // number of edges
        uint32_t *m_hIncrDegrees;                   // pinned helper array for BFS degrees scan
        std::vector<char> m_hIsolatedVertices;      // whether the vertex is isolated

        std::vector<Cluster> m_hClusters;           // detected clusters
        std::vector<int32_t> m_hClusterInds;        // which cluster each vertex belongs to, -1 if none
        std::vector<float3> m_hClusterCentroids;    // clusters mean points
        bool *m_hClustersMerged;                    // whether any clusters will be merged

        // GPU data
        float3 *m_dPos;                             // particle positions
        float3 *m_dSortedPos;                       // sorted particle positions
        uint32_t *m_dContour;                       // contour array

        int32_t *m_dAdjTriangle;                    // upper half of adjacency matrix without main diagonal
        char *m_dIsolatedVertices;                  // whether the vertex is isolated

        int32_t *m_dClusterInds;                    // which cluster each vertex belongs to, -1 if none
        float3 *m_dClusterCentroids;                // clusters mean points
        uint32_t *m_dClusterCandidatesInds;         // 2 pairs of closest neighbours per each cluster centroid
        uint32_t *m_dClusterEdges;                  // selected edges between clusters

        uint32_t *m_dAdjacencyList;                 // adjacency list representation
        uint32_t *m_dEdgesOffset;                   // scanned vertices degrees
        uint32_t *m_dEdgesSize;                     // vertices degrees

        // Grid data for sorting method
        uint32_t *m_dGridParticleHash;              // grid hash value for each particle
        uint32_t *m_dGridParticleIndex;             // particle index for each particle
        uint32_t *m_dCellStart;                     // index of start of each cell in sorted list
        uint32_t *m_dCellEnd;                       // index of end of cell

        // Params
        KernelParams m_params;                      // constants for kernels
        StopWatchInterface *m_timer;                // timer
};

#endif // __PARTICLE_SYSTEM_H__
