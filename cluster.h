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

    uint longestPathVertices;       // length of longest path in vertices
    float longestPath;              // longest path length

    uint branchingsCount;           // number of vertices of degree > 2
    uint leavesCount;               // number of vertices of degree 1

    float3 massCentre;              // center of mass
};

#endif //__CLUSTER_H__