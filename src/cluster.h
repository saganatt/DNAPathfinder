#ifndef __CLUSTER_H__
#define __CLUSTER_H__

struct Cluster {
    uint32_t clusterSize;           // how many particles are in the cluster
    float3 centroid;                // mean point

    float longestEdge;              // longest edge length
    float shortestEdge;             // shortest edge length

    // These two are rather approximate values - can be accurate in case the cluster is a tree
    uint32_t longestPathVertices;   // length of longest path in vertices
    float longestPath;              // longest path length

    uint32_t branchingsCount;       // number of vertices of degree > 2
};

#endif //__CLUSTER_H__