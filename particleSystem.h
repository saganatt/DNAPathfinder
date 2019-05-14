#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0

#include <helper_functions.h>
#include <stdint.h>
#include "particles_kernel.cuh"
#include "vector_functions.h"

// Particle system class
class ParticleSystem
{
    public:
        ParticleSystem(uint numParticles, uint3 gridSize, float *particles, uint32_t *p_indices,
                float searchRadius, uint32_t *contour, uint3 contourSize, float3 voxelSize,
                float maxCoordinate);
        ~ParticleSystem();

        void update();

        float *getPositions();
        void   setPositions(const float *position, int start, int count);

        int32_t *getAdjTriangle();
        int32_t getAdjTriangleEntry(int32_t *adjTriangle, uint32_t row, uint32_t column);
        void setAdjTriangleEntry(int32_t *adjTriangle, uint32_t row, uint32_t column, int32_t value);
        int32_t *getPairsInd();

        int getNumParticles() const
        {
            return m_numParticles;
        }

        uint32_t getAdjTriangleSize() const
        {
            return m_adjTriangleSize;
        }

        uint32_t *getContour();
        void setContour(const uint32_t *p_contour, int start, int count);
        uint3 getContourSize() const
        {
            return m_contourSize;
        }
        float3 getVoxelSize() const
        {
            return m_params.voxelSize;
        }

        void dumpGrid();
        void dumpParticles(uint start, uint count);

        uint3 getGridSize()
        {
            return m_params.gridSize;
        }
        float3 getWorldOrigin()
        {
            return m_params.worldOrigin;
        }
        float3 getCellSize()
        {
            return m_params.cellSize;
        }

    protected: // methods
        //ParticleSystem() {}

        void _initialize();
        void _finalize();

    protected: // data
        bool m_bInitialized;
        uint m_numParticles;

        // CPU data
        float *m_hPos;              // particle positions

        int32_t *m_hAdjTriangle;         // less than half of adjacency matrix
        uint32_t m_adjTriangleSize;

        uint32_t *m_hContour;  // values from contour matrix
        uint3 m_contourSize; // matrix sizes in each dimension
        // Contour access at (x, y, z) (starting from 0): x + y * contourSize.x + z * contourSize.x * contourSize.y;
        // Multiply each coordinate by voxelSize to get world coordinates of given matrix cell

        uint  *m_hParticleHash;
        uint  *m_hCellStart;
        uint  *m_hCellEnd;

        // GPU data
        float *m_dPos;
        float *m_dSortedPos;

        int32_t *m_dAdjTriangle;

        uint32_t *m_dContour;

        // grid data for sorting method
        uint  *m_dGridParticleHash; // grid hash value for each particle
        uint  *m_dGridParticleIndex;// particle index for each particle
        uint  *m_dCellStart;        // index of start of each cell in sorted list
        uint  *m_dCellEnd;          // index of end of cell

        uint   m_gridSortBits;

        // params
        SimParams m_params;
        uint3 m_gridSize;
        uint m_numGridCells;

        StopWatchInterface *m_timer;
};

#endif // __PARTICLESYSTEM_H__
