#ifndef __IO_FUNCTIONS_H__
#define __IO_FUNCTIONS_H__

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <ios>
#include <cnpy.h>

//#include <unordered_set>
//#include "flat_hash_map.hpp"

#include "particleSystem.h"

const float maxCoordinate = 1.0f;//10000.0f;
std::string defaultScript = "script.dp";
std::string defaultContourFile = "data/nuc9-syg2v2.npy";

uint32_t numParticles = 0;
ParticleSystem *psystem = 0;

// TODO: Hardcoded constant? Can it be determined somehow?
const float3 voxelSize = make_float3(7.00000014f, 7.00000014f, 7.00280101f);

void getParticlesData(char *file, float* particles[], uint32_t* p_indices[]);

//ska::flat_hash_set<std::string> removeDuplicatePairs(uint32_t *p_indices, uint32_t *pairsInd);
std::string composeChimeraBondCommand(uint32_t ind1, uint32_t ind2);
//void writeChimeraScript(ska::flat_hash_set<std::string> pairsSet, std::string script);
void writeChimeraScript(int32_t *pairsInd, std::string script);

void getContourMatrix(uint32_t *contour[], uint3 *contourSize, std::string matrixFile);

void getParticlesData(char *file, float* particles[], uint32_t* p_indices[]) {
    std::ifstream ifile(file);
    if(!ifile) {
        printf("Could not open the file for reading.\n");
        exit(EXIT_FAILURE);
    }
    printf("Reading from %s...\n", file);
    numParticles = (uint32_t)std::count(std::istreambuf_iterator<char>(ifile),
                                        std::istreambuf_iterator<char>(), '\n');
    *particles = new float[3 * numParticles];
    *p_indices = new uint32_t[numParticles];
    std::string line;
    ifile.clear();
    ifile.seekg(0, std::ios::beg);
    for(int i = 0; i < numParticles; i++) {
        if(std::getline(ifile, line)) {
            std::istringstream in(line);
            float x, y, z;
            in >> (*p_indices)[i] >> x >> y >> z;
            (*particles)[3*i] = x / maxCoordinate;
            (*particles)[3*i+1] = y / maxCoordinate;
            (*particles)[3*i+2] = z / maxCoordinate;
            if(!in) {
                printf("Unexpected error when reading from a file.\n");
                ifile.close();
                exit(EXIT_FAILURE);
            }
        }
        else {
            printf("Unexpected error when reading from a file.\n");
            ifile.close();
            exit(EXIT_FAILURE);
        }
    }
    ifile.close();
    printf("Succesfully read all particles data\n\n");
}

/*ska::flat_hash_set<std::string> removeDuplicatePairs(uint32_t *p_indices, uint32_t *pairsInd) {
    ska::flat_hash_set<std::string> pairsSet;
    int n = psystem->getNumPairs();
    for(int i = 0; i < n; i++) {
        // a bit of sorting to avoid duplicate inverted pairs
        if(pairsInd[2*i] < p_indices[i]) {
            pairsSet.insert(composeChimeraBondCommand(pairsInd[2*i], p_indices[i]));
        }
        else {
            pairsSet.insert(composeChimeraBondCommand(p_indices[i], pairsInd[2*i]));
        }
        if(pairsInd[2*i+1] < p_indices[i]) {
            pairsSet.insert(composeChimeraBondCommand(pairsInd[2*i+1], p_indices[i]));
        }
        else {
            pairsSet.insert(composeChimeraBondCommand(p_indices[i], pairsInd[2*i+1]));
        }
    }
    return pairsSet;
}*/

std::string composeChimeraBondCommand(uint32_t ind1, uint32_t ind2) {
    if(ind1 == 0 || ind2 == 0) {
#if DEBUG
        printf("Skipping empty pair\n");
#endif
        return "";
    }
    std::ostringstream stm;
    stm << "#0:" << ind1 << " #0:" << ind2;
    return stm.str() ;
}


void writeChimeraScript(int32_t *pairsInd, std::string script) {
    std::ofstream ofile(script);
    if(!ofile) {
        printf("Could not open the file for writing.\n");
        exit(EXIT_FAILURE);
    }

    std::cout << "Writing new script to " << script << "..." << std::endl;
    //ofile << "# Generated file to show particle bonds in Chimera" << std::endl << std::endl;

    for(int i = 0; i < numParticles; i++) {
        if(pairsInd[i] > 0) {
            ofile << composeChimeraBondCommand(i + 1, pairsInd[i] + 1) << std::endl;
        }
    }
/*
    for(std::string s : pairsSet) {
        ofile << s << std::endl;
    }
*/

    //ofile << std::endl << "# End of generated file" << std::endl;
    printf("Chimera script generated successfully\n\n");

    ofile.close();
}


void getContourMatrix(uint32_t *contour[], uint3 *contourSize, std::string matrixFile) {
    std::cout << "Loading contour matrix from: " << matrixFile << std::endl;

    cnpy::NpyArray array = cnpy::npy_load(matrixFile);
    assert(array.word_size == sizeof(uint32_t)); // actually the array is uint32_t
    assert(array.shape.size() == 3);

    uint32_t *temp = array.data<uint32_t>();
    *contourSize = make_uint3((uint)array.shape[2], (uint)array.shape[1], (uint)array.shape[0]);

    *contour = new uint32_t[contourSize->x * contourSize->y * contourSize->z];
    for(int i = 0; i < contourSize->x * contourSize->y * contourSize->z; i++) {
        (*contour)[i] = temp[i];
    }

    std::cout << "Contour size: " << contourSize->x << ", " << contourSize->y << ", " << contourSize->z << std::endl;

    //int start = 62 * contourSize->y * contourSize->x + 77 * contourSize->x + 107;
    //std::cout << "Checking 1451 at [107, 77, 62] = [62 * cs.y * cs.x + 77 * cs.x + 107 = " << start << std::endl;
    //for(int i = start; i < start + contourSize->x; i++) {
    //    std::cout << (*contour)[i] << " ";
    //}
    //std::cout << std::endl;

    std::cout << "Succesfully read all contour data" << std::endl << std::endl;
}

#endif


