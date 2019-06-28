# DNAPathfinder

<h2>TSP variant on genes</h2>

The aim: find approximate shortest path that joins all atoms and does not go outside the contour given.

The program calculates shortest paths between pairs of gene atoms, within the range of the search radius provided as an argument. Next, it finds clusters of connected atoms and connects them in a loop until there is only one cluster left or loops count is bigger than the maximum value given as parameter.

<h3>Usage</h3>

1. *.cmap data should be converted into *.npy.<br>
2. Main program compilation with make. Available options:<br>
  ```
  make print_dbg={1, 2, 3}	  compile with different prints verbosity
                              1 - save current clusters to files in each program loop
                              2 - 1 + measure time and print loop info after each loop
                              3 - 2 + print particles outside the contour and adjacency list on each creation
  make dbg=1                  compile in CUDA debug mode (-g -G) and print_dbg=2
  ```
3. Launching the main program:<br>
  ```
  ./particles [OPTION [=VALUE]]
  ```
  Available options:<br>
  ```
  -g    set grid size, default:                                                                64
  -r    set search radius for particles pairs, default:                                        10
  -cr   set search radius for clusters pairs, default:                                         20
  -l    set max number of main loops to execute, 0 means infinite, default:                    0
  -f    specifies path to atoms *.pdb file
  -cf   specifies path to *.npy contour file
  -s    specifies prefix of output *.dp script for Chimera PseudoBond Reader, default:         ../data/results/all
  -t    specifies prefix of output *.csv file with clusters statistics, default:               ../data/results/clusters
  -p    specifies prefix of output *.pdb file with clusters mass centres, default:             ../data/results/clusters
  -cs   specifies prefix of output *.dp clusters script for Chimera PseudoBond Reader, default:../data/results/clusters
  -h, --help  display this help and exit
  ```
  4. To see the results, you can launch Chimera UCSF (https://www.cgl.ucsf.edu/chimera/) like this:<br>
  ```
    chimera <pdb_file> <cmap_file>
  ```
   And then use Tools -> Depiction -> PseudoBond Reader to read generated \*.dp file(s). (\*.dp is not a real extension.)<br>
   Bonds from different clusters are marked with different colours.<br>
   For better visibility, you might want to set atoms to spheres / set the contour to mesh.

<h3>Used codes</h3>

1. Atoms representation in world space is based on Nvidia's particles sample.<br>
2. Cluster statistics use variants of reduce algorithm from Nvidia's reduce sample and Mark Harris' code at: http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf<br>
3. Bresenham algorithm is adapted from Will Navidson's (yamamushi) gist at: https://gist.github.com/yamamushi/5823518#file-bresenham3d<br>
4. Breadth-First Search is adjusted from: https://github.com/rafalk342/bfs-cuda<br>
