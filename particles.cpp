// CUDA runtime
#include <cuda_runtime.h>

// CUDA utilities and system includes
//#include <helper_functions.h>
//#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_string.h>

// Includes
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <chrono>

#include "particleSystem.h"
#include "io_functions.h"

#define DEBUG 0

const uint32_t defaultGridSize = 64;
const float defaultSearchRadius = 16.0f / maxCoordinate; // 10 x atom size

uint3 gridSize;

StopWatchInterface *timer = NULL;

extern "C" void cudaInit(int argc, char **argv);

void usage(const char* name);
void parseArguments(int argc, char **argv, char **file, std::string *script, uint32_t* gridDim, float* searchRadius,
        std::string *contourFile);

void initParticleSystem(uint3 gridSize, float *particles, uint32_t *p_indices, float searchRadius,
        uint32_t *contour, uint3 contourSize, float3 voxelSize);
void cleanup();
void runSystem();

void initParticleSystem(uint3 gridSize, float *particles, uint32_t *p_indices, float searchRadius,
        uint32_t *contour, uint3 contourSize, float3 voxelSize)
{
    psystem = new ParticleSystem(numParticles, gridSize, particles, p_indices, searchRadius,
            contour, contourSize, voxelSize, maxCoordinate);

    sdkCreateTimer(&timer);
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    if (psystem)
    {
        delete psystem;
    }
}

void runSystem()
{
    printf("Run %u particles...\n\n", numParticles);

    cudaDeviceSynchronize();
    sdkStartTimer(&timer);

    psystem->update();

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    float fAvgSeconds = ((float)1.0e-3 * (float)sdkGetTimerValue(&timer));

    printf("DNAPathfinder, Throughput = %.4f KParticles/s, Time = %.5f s, Size = %u particles, NumDevsUsed = %u, Workgroup = %u\n\n", (1.0e-3 * numParticles)/fAvgSeconds, fAvgSeconds, numParticles, 1, 0);

}

int main(int argc, char **argv)
{
    printf("Starting DNA Pathfinder...\n\n");

    uint32_t gridDim = defaultGridSize;
    char *file;
    float searchRadius = defaultSearchRadius;
    std::string script(defaultScript);
    std::string contourFile(defaultContourFile);

    parseArguments(argc, argv, &file, &script, &gridDim, &searchRadius, &contourFile);

    gridSize.x = gridSize.y = gridSize.z = gridDim;
    printf("Grid: %d x %d x %d = %d cells\n", gridSize.x, gridSize.y, gridSize.z, gridSize.x*gridSize.y*gridSize.z);
    printf("Search radius: %f\n", searchRadius);
    std::cout << "Output script will be saved in: " << script << std::endl << std::endl;

    float *particles;
    uint32_t *p_indices;
    getParticlesData(file, &particles, &p_indices);

    uint32_t *contour;
    uint3 contourSize;
    getContourMatrix(&contour, &contourSize, contourFile);

#if DEBUG
    for(int i = 0; i < numParticles; i++) {
        std::cout << p_indices[i] << " " << p_types[i] << " ";
        std::cout << particles[3*i] << " " << particles[3*i+1] << " " << particles[3*i+2] << std::endl;
    }
    std::cout << std::endl;
#endif

    cudaInit(argc, argv);

    initParticleSystem(gridSize, particles, p_indices, searchRadius, contour, contourSize, voxelSize);

    runSystem();

    int32_t *pairsInd = psystem->getPairsInd();

#if DEBUG
    psystem->dumpParticles(0, numParticles);
    std::cout << std::endl;
    std::cout << "Printing pairs" << std::endl;
    for(int i = 0; i < psystem->getNumPairs(); i++) {
        std::cout << pairsInd[2*i] << ", " << pairsInd[2*i+1] << std::endl;
    }
    std::cout << std::endl;
#endif

    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    //ska::flat_hash_set<std::string> pairsSet = removeDuplicatePairs(p_indices, pairsInd);
    //writeChimeraScript(pairsSet, script);

    writeChimeraScript(pairsInd, script);

    sdkStopTimer(&timer);
    float fAvgSeconds = ((float)1.0e-3 * (float)sdkGetTimerValue(&timer));

    printf("DNAPathfinder writing Chimera script, Throughput = %.4f KParticles/s, Time = %.5f s, Size = %u particles, NumDevsUsed = %u, Workgroup = %u\n\n", (1.0e-3 * numParticles)/fAvgSeconds, fAvgSeconds, numParticles, 1, 0);

    delete [] pairsInd;
    cleanup();

    exit(EXIT_SUCCESS);
}

void parseArguments(int argc, char **argv, char **file, std::string *script,
        uint32_t* gridDim, float* searchRadius, std::string *contourFile) {
    if (argc > 1) {
        if (checkCmdLineFlag(argc, (const char **) argv, "help") ||
            checkCmdLineFlag(argc, (const char **) argv, "h")) {
            usage(argv[0]);
        }

        if (checkCmdLineFlag(argc, (const char **) argv, "f")) {
            if(!getCmdLineArgumentString(argc, (const char **) argv, "f", file)) {
                printf("Could not parse filename string.\n");
                usage(argv[0]);
            }
        }
        else if (checkCmdLineFlag(argc, (const char **) argv, "file")) {
            if(!getCmdLineArgumentString(argc, (const char **) argv, "file", file)) {
                printf("Could not parse filename string.\n");
                usage(argv[0]);
            }
        }
        else {
            printf("No particles data provided.\n");
            usage(argv[0]);
        }

        if (checkCmdLineFlag(argc, (const char **) argv, "grid")) {
            *gridDim = getCmdLineArgumentInt(argc, (const char **) argv, "grid");
        }
        else if (checkCmdLineFlag(argc, (const char **) argv, "g")) {
            *gridDim = getCmdLineArgumentInt(argc, (const char **) argv, "g");
        }

        if (checkCmdLineFlag(argc, (const char **) argv, "r")) {
            *searchRadius = getCmdLineArgumentFloat(argc, (const char **) argv, "r");
        }

        if (checkCmdLineFlag(argc, (const char **) argv, "s")) {
            char *ch;
            if(!getCmdLineArgumentString(argc, (const char **) argv, "s", &ch)) {
                printf("Could not parse script filename string.\n");
                usage(argv[0]);
            }
            script->assign(ch);
        }

        if (checkCmdLineFlag(argc, (const char **) argv, "c")) {
            char *ch;
            if(!getCmdLineArgumentString(argc, (const char **) argv, "c", &ch)) {
                printf("Could not parse contour filename string.\n");
                usage(argv[0]);
            }
            contourFile->assign(ch);
        }
    }
    else {
        printf("No particles data provided.\n");
        usage(argv[0]);
    }
}

void usage(const char* name) {
    printf("Usage: %s [OPTION [=VALUE]]\n", name);
    printf("Available options:\n");
    printf("-g, --grid\tset grid size, default:\t\t\t\t\t\t\t\t%u\n", defaultGridSize);
    printf("-f, --file\tspecifies path to the file with particles coordinates\n");
    printf("-s, \t\tspecifies path to output script for Chimera PseudoBond Reader, default:\t%s\n",
        defaultScript.c_str());
    printf("-c, \t\tspecifies path to .npy contour file, default:\t\t\t\t\t%s\n", defaultContourFile.c_str());
    printf("-r, \t\tset search radius for particles pairs, default:\t\t\t\t\t%f\n", defaultSearchRadius);
    printf("-h, --help\tdisplay this help and exit\n");
    exit(EXIT_FAILURE);
}
