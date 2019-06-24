#include "filesIO.h"

std::string script;
std::string clustersScript;
std::string clustersCsvFile;
std::string clustersPdbFile;

int scriptsCount = 1;
int clustersScriptsCount = 1;
int clustersCsvFilesCount = 1;
int clustersPdbFilesCount = 1;

void getParticlesData(std::vector<float3> &particles, std::vector<uint32_t> &p_indices,
                      uint32_t &numParticles, const std::string &file) {
    std::ifstream ifile(file);
    if (!ifile) {
        throw std::runtime_error("Could not open the file for reading.");
    }
    std::cout << "Reading from " << file << "..." << std::endl;
    std::string line;
    ifile.clear();
    ifile.seekg(0, std::ios::beg);
    while (std::getline(ifile, line)) {
        std::istringstream in(line);
        float x, y, z;
        uint32_t ind;
        in.ignore(6);
        if (!in) {
            throw std::runtime_error("Unexpected error when reading from a file.");
        }
        in >> ind;
        if (!in) {
            throw std::runtime_error("Unexpected error when reading from a file.");
        }
        in.ignore(19);
        if (!in) {
            throw std::runtime_error("Unexpected error when reading from a file.");
        }
        in >> x >> y >> z;
        if (!in) {
            throw std::runtime_error("Unexpected error when reading from a file.");
        }
        p_indices.emplace_back(ind);
        particles.emplace_back(make_float3(x, y, z));
        numParticles++;
    }
    std::cout << "Particles data read successfully." << std::endl << std::endl;
}

void getContourMatrix(std::vector<uint32_t> &contour, uint3 &contourSize, const std::string &matrixFile) {
    std::cout << "Loading contour matrix from: " << matrixFile << std::endl;

    cnpy::NpyArray array = cnpy::npy_load(matrixFile);
    assert(array.word_size == sizeof(uint32_t));
    assert(array.shape.size() == 3);

    uint32_t *temp = array.data<uint32_t>();
    contourSize = make_uint3((uint32_t)array.shape[2], (uint32_t)array.shape[1], (uint32_t)array.shape[0]);
    contour = std::vector<uint32_t>(temp, temp + (contourSize.x * contourSize.y * contourSize.z));

    std::cout << "Contour size: "
              << contourSize.x << ", " << contourSize.y << ", " << contourSize.z << std::endl;

    std::cout << "Succesfully read all contour data" << std::endl << std::endl;
}

std::string composeChimeraBondCommand(ColorGenerator &colorGen, uint32_t ind1, uint32_t ind2,
                                      uint32_t segment1, uint32_t segment2, uint32_t clusterInd) {
    if (ind1 == 0 || ind2 == 0) {
        return "";
    }
    std::ostringstream stm;
    stm << "#" << segment1 << ":" << ind1 << " #" << segment2 << ":" << ind2
        << " " << colorGen.getNextColor(clusterInd);
    return stm.str();
}

void writeChimeraScriptFromAdjList(const std::vector<uint32_t> &adjList,
                                   const std::vector<uint32_t> &edgesOffset,
                                   const std::vector<uint32_t> &edgesSize,
                                   const std::vector<int32_t> &clusterInds,
                                   const uint32_t &numParticles,
                                   ColorGenerator &colorGen, bool suffix) {
    std::ostringstream stm;
    stm << script;
    if (suffix) {
        stm << "_" << scriptsCount;
        scriptsCount++;
    }
    stm << ".dp";
    std::string scriptPath = stm.str();
    std::ofstream ofile(scriptPath);
    if (!ofile) {
        throw std::runtime_error("Could not open the file for writing.");
    }

    std::cout << "Writing new script to " << scriptPath << "..." << std::endl;

    for (int32_t i = 0; i < numParticles; i++) {
        for (int32_t j = edgesOffset[i]; j < edgesOffset[i] + edgesSize[i]; j++) {
            if (adjList[j] > i) {
                ofile << composeChimeraBondCommand(colorGen, i + 1, adjList[j] + 1, 0, 0, clusterInds[i])
                      << std::endl;
            }
        }
    }

    std::cout << "Chimera script generated successfully." << std::endl << std::endl;

    ofile.close();
}

void writeChimeraScriptFromClustersCandidatesList(const std::vector<uint32_t> &pairsInd, uint32_t clustersCount,
                                                  const std::vector<int32_t> &clusterInds,
                                                  const uint32_t &numParticles,
                                                  ColorGenerator &colorGen, bool suffix) {
    std::ostringstream stm;
    stm << clustersScript;
    if (suffix) {
        stm << "_" << clustersScriptsCount;
        clustersScriptsCount++;
    }
    stm << ".dp";
    std::string clustersScriptPath = stm.str();
    std::ofstream ofile(clustersScriptPath);
    if (!ofile) {
        throw std::runtime_error("Could not open the file for writing.");
    }

    std::cout << "Writing new clusters script to " << clustersScriptPath << "..." << std::endl;

    for (int32_t i = 0; i < clustersCount; i++) {
        for (int32_t j = 0; j < 4; j++) {
            int32_t ind = pairsInd[4 * i + j];
            if (ind < numParticles) {
                ofile << composeChimeraBondCommand(colorGen, i + 1, ind + 1, 1, 0, clusterInds[ind])
                      << std::endl;
            }
        }
    }

    std::cout << "Chimera clusters script generated successfully." << std::endl << std::endl;

    ofile.close();
}

void writeClustersStatsToCsv(const std::vector<Cluster> &clusters, bool suffix) {
    std::ostringstream stm;
    stm << clustersCsvFile;
    if (suffix) {
        stm << "_" << clustersCsvFilesCount;
        clustersCsvFilesCount++;
    }
    stm << ".csv";
    std::string clustersCsvFilePath = stm.str();
    std::ofstream ofile(clustersCsvFilePath);
    if (!ofile) {
        throw std::runtime_error("Could not open the file for writing.");
    }

    std::cout << "Saving clusters statistics to " << clustersCsvFilePath << "..." << std::endl;

    for (int32_t i = 0; i < clusters.size(); i++) {
        ofile << i << "," << clusters[i].clusterSize
              << "," << clusters[i].shortestEdge << "," << clusters[i].longestEdge
              << "," << clusters[i].longestPath << "," << clusters[i].longestPathVertices
              << "," << clusters[i].branchingsCount
              << "," << clusters[i].centroid.x << "," << clusters[i].centroid.y
              << "," << clusters[i].centroid.z
              << std::endl;
    }

    std::cout << "Clusters statistics saved successfully" << std::endl << std::endl;

    ofile.close();
}

// Simplified record with redundant parts hardcoded
std::string composePdbAtomRecord(uint32_t index, float3 pos) {
    std::ostringstream stm;

    // Simplified format:  ATOM {index} B   BEA A {index} {pos.x} {pos.y} {pos.z}  0.00  0.00           C
    // Example:            ATOM      1  B   BEA A   1     609.000 567.000 308.123  0.00  0.00           C

    stm << std::setfill(' ') << std::right << std::fixed << std::showpoint;
    stm << "ATOM  ";
    stm << std::setw(5) << index;
    stm << "  B   BEA A";
    stm << std::setw(4) << index;
    stm << "    ";
    stm << std::setprecision(3) << std::setw(8) << pos.x;
    stm << std::setprecision(3) << std::setw(8) << pos.y;
    stm << std::setprecision(3) << std::setw(8) << pos.z;
    stm << "  0.00  0.00           C";

    return stm.str();
}

void writeClustersCentroidsToPdb(const std::vector<float3> &centroids, bool suffix) {
    if (centroids.size() > 9999) {
        throw std::runtime_error("Too many clusters for a single PDB file, exiting...");
    }

    std::ostringstream stm;
    stm << clustersPdbFile;
    if (suffix) {
        stm << "_" << clustersPdbFilesCount;
        clustersPdbFilesCount++;
    }
    stm << ".pdb";
    std::string clustersPdbFilePath = stm.str();
    std::ofstream ofile(clustersPdbFilePath);
    if (!ofile) {
        throw std::runtime_error("Could not open the file for writing.");
    }

    std::cout << "Writing clusters mass centres to " << clustersPdbFilePath << "..." << std::endl;

    for (int32_t i = 0; i < centroids.size(); i++) {
        ofile << composePdbAtomRecord(i + 1, centroids[i]) << std::endl;
    }

    std::cout << "PDB clusters file generated successfully." << std::endl << std::endl;

    ofile.close();
}