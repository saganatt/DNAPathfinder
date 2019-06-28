#ifndef __FILES_IO_H__
#define __FILES_IO_H__

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <ios>
#include <cnpy.h>

#include <helper_timer.h>

#include "particleSystem.h"
#include "colorGenerator.h"

const std::string defaultScript = "../data/results/all";
const std::string defaultClustersScript = "../data/results/clusters";
const std::string defaultClustersCsvFile = "../data/results/clusters";
const std::string defaultClustersPdbFile = "../data/results/clusters";

extern std::string script;
extern std::string clustersScript;
extern std::string clustersCsvFile;
extern std::string clustersPdbFile;

extern int scriptsCount;
extern int clustersScriptsCount;
extern int clustersCsvFilesCount;
extern int clustersPdbFilesCount;

void getParticlesData(std::vector<float3> &particles, std::vector<uint32_t> &p_indices,
                      uint32_t &numParticles, const std::string &file);

void getContourMatrix(std::vector<uint32_t> &contour, uint3 &contourSize, const std::string &matrixFile);

std::string composeChimeraBondCommand(ColorGenerator &colorGen, uint32_t ind1, uint32_t ind2,
                                      uint32_t segment1 = 0, uint32_t segment2 = 0,
                                      uint32_t clusterInd = 0);

void writeChimeraScriptFromAdjList(const std::vector<uint32_t> &adjList,
                                   const std::vector<uint32_t> &edgesOffset,
                                   const std::vector<uint32_t> &edgesSize,
                                   const std::vector<int32_t> &clusterInds,
                                   const uint32_t &numParticles,
                                   ColorGenerator &colorGen, bool suffix = true);

void writeChimeraScriptFromClustersCandidatesList(const std::vector<uint32_t> &pairsInd, uint32_t clustersCount,
                                                  const std::vector<int32_t> &clusterInds,
                                                  const uint32_t &numParticles,
                                                  ColorGenerator &colorGen, bool suffix = true);

void writeClustersStatsToCsv(const std::vector<Cluster> &clusters, bool suffix = true);

std::string composePdbAtomRecord(uint32_t index, float3 pos);

void writeClustersCentroidsToPdb(const std::vector<float3> &centroids, bool suffix = true);

#endif // __FILES_IO_H__