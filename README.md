# DNAPathfinder (WIP)

<h2>TSP variant on genes</h2>

The aim: find approximate shortest path that joins all atoms and does not go outside the contour given.

Based on NVIDIA particles representation. The program calculates shortest paths between pairs of gene atoms, within the range of the search radius provided as an argument.

<i>(In progress)</i> Detecting clusters of atoms and determining their size, longest path length etc. and how it depends on the search radius.<br>
<i>(In planning)</i> Aggregating clusters into points and further connecting them - until all clusters / atoms are connected.

<h3>Usage</h3>

1. Preprocess *.pdb file with atom data: ` ./data/extract_ind_coos.sh <pdb_file> <new_file>` <br>
2. *.cmap data should be converted into *.npy.<br>
3. Main program compilation with make.<br>
  <b>NOTE 1: CMakeLists.txt are used only for tweaking CLion editor, use only Makefiles for correct compilation</b><br>
  <b>NOTE 2: 2 sample Makefiles are provided - still you may need to adjust CUDA files path / gencode arguments</b><br>
4. Launching the main program:<br>
  ```
  ./particles [OPTION [=VALUE]]
  ```
  Available options:<br>
  ```
  -g, --grid	set grid size, default:						                       64
  -f, --file	specifies path to the file with particles coordinates
  -s, 		specifies path to output .dp script for Chimera PseudoBond Reader, default:	script.dp
  -c, 		specifies path to .npy contour file, default:			     data/nuc9-syg2v2.npy
  -r, 		set search radius for particles pairs, default:			                16.000000
  -h, --help	display this help and exit
  ```
  5. To see the results, you can launch Chimera UCSF (https://www.cgl.ucsf.edu/chimera/) like this:<br>
  ```
    chimera <pdb_file> <cmap_file>
  ```
   And then use Tools -> Depiction -> PseudoBond Reader to read generated \*.dp file(s).<br>
   For better visibility, you might want to change colours / set atoms to spheres / set the contour to mesh.
