#ifndef __CLUSTER_H__
#define __CLUSTER_H__

typedef unsigned int uint;

struct Cluster
{
    float3 minExtremes;             // particles of min x, y, z
    int3 minExtremesInd;            // particles indices in particle system
    float3 maxExtremes;             // particles of max x, y, z
    int3 maxExtremesInd;            // particles indices in particle system

    uint clusterSize;               // how many particles are in the cluster

    float longestEdge;              // longest edge length
    float shortestEdge;             // shortest edge length
    //float avEdge;                   // average edge length

    uint longestPathVertices;       // length of longest path in vertices
    float longestPath;              // longest path length

    uint branchingsCount;           // number of vertices of degree > 2
    //float3 *leaves;                  // vertices of degree 1
    //int32_t *leavesInd;             // indices of leaves in particle system
    uint leavesCount;               // number of vertices of degree 1
};

#endif //__CLUSTER_H__