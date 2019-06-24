#include <cuda_runtime.h>
#include <helper_string.h>
#include <helper_cuda.h>

#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <exception>

#include "particleSystem.h"
#include "filesIO.h"

const uint32_t defaultGridSize = 64;
const float defaultSearchRadius = 16.0f; // 10 x atom size
// Contour unit of space
const float3 voxelSize = make_float3(7.00000014f, 7.00000014f, 7.00280101f);
StopWatchInterface *timer = nullptr;

extern "C" void cudaInit(int32_t argc, char **argv);

void runSystem(ParticleSystem &psystem, const uint32_t &numParticles);

void usage(const char *name);

void parseArguments(int32_t argc, char **argv, std::string &file, std::string &contourFile,
                    uint32_t &gridDim, float &searchRadius, float &clustersSearchRadius);

int32_t main(int argc, char **argv) {
    std::cout << "Starting DNA Pathfinder..." << std::endl << std::endl;

    uint32_t numParticles = 0;
    uint32_t gridDim = defaultGridSize;
    float searchRadius = defaultSearchRadius;
    float clustersSearchRadius = 2 * defaultSearchRadius;
    std::string file;
    std::string contourFile = std::string(defaultContourFile);
    script = std::string(defaultScript);
    clustersScript = std::string(defaultClustersScript);
    clustersCsvFile = std::string(defaultClustersCsvFile);
    clustersPdbFile = std::string(defaultClustersPdbFile);

    try {
        parseArguments(argc, argv, file, contourFile, gridDim, searchRadius, clustersSearchRadius);
    }
    catch (std::runtime_error &error) {
        std::cout << error.what() << std::endl;
        return EXIT_FAILURE;
    }

    uint3 gridSize = make_uint3(gridDim, gridDim, gridDim);
    std::cout << "Grid: " << gridSize.x << " x " << gridSize.y << " x " << gridSize.z
              << " = " << gridSize.x * gridSize.y * gridSize.z << " cells" << std::endl;
    std::cout << "Search radius: " << searchRadius << std::endl;
    std::cout << "Clusters search radius: " << clustersSearchRadius << std::endl;
    std::cout << "Output script will be saved in: " << script << "*.dp" << std::endl;
    std::cout << "Clusters statistics will be saved in: " << clustersCsvFile << "*.csv" << std::endl;
    std::cout << "Clusters mass centres will be saved in: " << clustersPdbFile << "*.pdb" <<  std::endl;
    std::cout << "Output clusters script will be saved in: " << clustersScript << "*.dp" << std::endl << std::endl;

    std::vector<float3> particles;
    std::vector<uint32_t> p_indices;
    try {
        getParticlesData(particles, p_indices, numParticles, file);
    }
    catch (std::runtime_error &error) {
        std::cout << error.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<uint32_t> contour;
    uint3 contourSize;
    try {
        getContourMatrix(contour, contourSize, contourFile);
    }
    catch (std::runtime_error &error) {
        std::cout << error.what() << std::endl;
        return EXIT_FAILURE;
    }

    cudaInit(argc, argv);
    ParticleSystem psystem(numParticles, gridSize, particles, p_indices, searchRadius,
                           clustersSearchRadius, contour, contourSize, voxelSize);
    sdkCreateTimer(&timer);
    runSystem(psystem, numParticles);

    std::vector<Cluster> clusters = psystem.getClusters();
    std::vector<int32_t> clusterInds = psystem.getClusterInds();
    std::vector<float3> clusterCentroids = psystem.getClusterCentroids();
    std::vector<uint32_t> clusterCandidatesInd = psystem.getClusterCandidatesInd();
    std::vector<uint32_t> edgesOffset = psystem.getEdgesOffset();
    std::vector<uint32_t> edgesSize = psystem.getEdgesSize();
    std::vector<uint32_t> adjList = psystem.getAdjList();

    ColorGenerator colorGen(clusters.size());
    try {
        writeChimeraScriptFromAdjList(adjList, edgesOffset, edgesSize, clusterInds,
                                      numParticles, colorGen, false);
        writeChimeraScriptFromClustersCandidatesList(clusterCandidatesInd, clusters.size(),
                                                     clusterInds, numParticles, colorGen, false);
        writeClustersCentroidsToPdb(clusterCentroids, false);
        writeClustersStatsToCsv(clusters, false);
    }
    catch (std::runtime_error &error) {
        std::cout << error.what() << std::endl;
        return EXIT_FAILURE;
    }

    sdkDeleteTimer(&timer);

    return EXIT_SUCCESS;
}

void runSystem(ParticleSystem &psystem, const uint32_t &numParticles) {
    std::cout << "Run " << numParticles << " particles..." << std::endl << std::endl;

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStartTimer(&timer);

    psystem.update();

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    float fAvgSeconds = ((float)1.0e-3 * (float)sdkGetTimerValue(&timer));

    std::cout << "DNAPathfinder, Throughput = " << std::setprecision(4) << (1.0e-3 * numParticles) / fAvgSeconds
              << " KParticles/s, Time = " << std::setprecision(5) << fAvgSeconds
              << " s, Size = " << numParticles << std::endl << std::endl;
}

void parseArguments(int32_t argc, char **argv, std::string &file, std::string &contourFile,
                    uint32_t &gridDim, float &searchRadius, float &clustersSearchRadius) {
    if (argc == 1) {
        std::cout << "No particles data provided." << std::endl;
        usage(argv[0]);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "h")) {
        usage(argv[0]);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "f")) {
        char *ch;
        if (!getCmdLineArgumentString(argc, (const char **)argv, "f", &ch)) {
            std::cout << "Could not parse data filename string." << std::endl;
            usage(argv[0]);
        }
        file.assign(ch);
    }
    else {
        std::cout << "No particles data provided." << std::endl;
        usage(argv[0]);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "g")) {
        gridDim = getCmdLineArgumentInt(argc, (const char **)argv, "g");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "r")) {
        searchRadius = getCmdLineArgumentFloat(argc, (const char **)argv, "r");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "cr")) {
        clustersSearchRadius = getCmdLineArgumentFloat(argc, (const char **)argv, "cr");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "s")) {
        char *ch;
        if (!getCmdLineArgumentString(argc, (const char **)argv, "s", &ch)) {
            std::cout << "Could not parse script filename string." << std::endl;
            usage(argv[0]);
        }
        script.assign(ch);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "l")) {
        char *ch;
        if (!getCmdLineArgumentString(argc, (const char **)argv, "l", &ch)) {
            std::cout << "Could not parse csv filename string." << std::endl;
            usage(argv[0]);
        }
        clustersCsvFile.assign(ch);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "p")) {
        char *ch;
        if (!getCmdLineArgumentString(argc, (const char **)argv, "p", &ch)) {
            std::cout << "Could not parse clusters PDB filename string." << std::endl;
            usage(argv[0]);
        }
        clustersPdbFile.assign(ch);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "cs")) {
        char *ch;
        if (!getCmdLineArgumentString(argc, (const char **)argv, "cs", &ch)) {
            std::cout << "Could not parse clusters script filename string." << std::endl;
            usage(argv[0]);
        }
        clustersScript.assign(ch);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "c")) {
        char *ch;
        if (!getCmdLineArgumentString(argc, (const char **)argv, "c", &ch)) {
            std::cout << "Could not parse contour filename string." << std::endl;
            usage(argv[0]);
        }
        contourFile.assign(ch);
        if (contourFile.substr(contourFile.length() - 4) != ".npy") {
            std::cout << "This is not a *.npy file name!" << std::endl;
            usage(argv[0]);
        }
    }
}

void usage(const char *name) {
    std::cout << "Usage: " << name << " [OPTION [=VALUE]]\n";
    std::cout << "Available options:\n";
    std::cout << "-g, \t\tset grid size, default:\t\t\t\t\t\t\t\t" << defaultGridSize << "\n";
    std::cout << "-f, \t\tspecifies path to the file with particles coordinates\n";
    std::cout << "-s, \t\tspecifies prefix of output script for Chimera PseudoBond Reader, default:\t\t"
              << defaultScript << "\n";
    std::cout << "-l, \t\tspecifies prefix of output *.csv file with clusters statistics, default:\t\t"
              << defaultClustersCsvFile << "\n";
    std::cout << "-p, \t\tspecifies prefix of output *.pdb file with clusters mass centres, default:\t"
              << defaultClustersPdbFile << "\n";
    std::cout << "-cs, \t\tspecifies prefix of output clusters script for Chimera PseudoBond Reader, default:"
              << defaultClustersScript << "\n";
    std::cout << "-c, \t\tspecifies path to .npy contour file, default:\t\t\t\t\t"
              << defaultContourFile << "\n";
    std::cout << "-r, \t\tset search radius for particles pairs, default:\t\t\t\t\t"
              << defaultSearchRadius << "\n";
    std::cout << "-cr, \t\tset search radius for clusters pairs, default:\t\t\t\t\t"
              << 2 * defaultSearchRadius << "\n";
    std::cout << "-h, --help\tdisplay this help and exit" << std::endl;
    throw std::runtime_error("");
}
