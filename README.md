# Persistent Cohomological Cycle Bootstrapping

## Outline 

This repository is forked from [interval-matching](https://github.com/inesgare/interval-matching), the code for the "Persistent Cohomological Cycle Matching" approach developed in the paper ["Fast Topological Signal Identification and Persistent Cohomological Cycle Matching" (García-Redondo, Monod, and Song 2022)](https://arxiv.org/abs/2209.15446). The original codes were co-written by [Inés García-Redondo](https://sites.google.com/view/ines-garcia-redondo/home) and [Anna Song](https://sites.google.com/view/annasong), performing cycle matching [1] while incorporating the computational advantages given by Ripser [2] and Ripser-image [3].

This fork is adapted to perform the analysis in [5] and is also included as a submodule of the corresponding [brain_representations](https://github.com/tyo8/brain_representations/tree/main) repository. 

### Structure of the repository

This repository is organised as follows. Each module listed below also has its own README(s) with more detailed information in corresponding directory.

#### `utils_match`
Prepare Ripser input, perform post-Ripser interval matching, and compute immediate derivative values from match data.

#### `visualization`
Visualize persistence and cycle-match output data, define and calculate quantities of post-hoc analysis.

#### `slurm_bash`
SLURM scripting at problem scale, including calls to Ripser.

#### `modified_ripser`
Ripser, but augmented with a line of code extracting its inbuilt lexicographical filtration refinement.

### Fork Contributions
This fork of [interval-matching](https://github.com/inesgare/interval-matching) contributes new functionality and makes significant changes to the code structure.
#### Functionality
- generalized to accept arbitrary (pre-computed) distance metrics in the bootstrapping case (see `create_ldm_images.py`)
- additional utilities for cycle registration over data bootstraps (see `create_ldm_images.py` and `generate_tag_subindex.py` in `utils_match`)
- integration of cycle matching with statistical permutation testing (see `comp_permtest_dists.py`, `bootstrap_dists.py`, `submit_(bsdist|permdist)_sbatch.sh`, and `(permtest|bootstrap)_distances.sh` in `visualization`)
- new modules for computation of Wasserstein variants (see `diagram_distances.py` in `visualization`)
- new visualizations of aggregated statistcs and distributions of bootstrapped cycles (see, e.g., `compare_topostats.py` and `distributional_summaries.py` in `visualization`)
- new toy examples (and corresponding manipulations) for validation and exploration (see `toy_models.py` in `visualization`)
#### Structure
- split cycle-matching implementation into modular structure (see `compute.py`, `extract.py`, and `match.py` in `utils_match`)
- re-factored script-style code into Pythonic functional programming style
- optimized for integration with high-performance computing (HPC) environment (including bash scripts with example calls: see `submit_*.sh` scripts in `slurm_bash`)
- re-organized directory structure

## Preparations

### C++

#### About C++

C++ is a general-purpose programming language which has object-oriented, generic, and functional features in addition to facilities for low-level memory manipulation. It is the language chosen for the codes Ripser [2] and Ripser-image [3]. The C++ files for those, with a slight modification needed to implement cycle matching, can be found in the folder `modified ripser`. 

#### Compiling C++ programs
Before running the code to perform cycle matching in this repository, one needs to compile the C++ files in the `modified ripser` folder. For that
- Install a C++ compiler in your computer. We recommend getting the compiler [GCC](https://gcc.gnu.org/).
	- *For Linux*: the default Ubuntu repositories contain a meta-package named build-essential that contains the GCC compiler and a lot of libraries and other utilities required for compiling software. You only need to run `sudo apt install build-essential` in a terminal to install it.
	- *For Windows*: you can install [Mingw-w64](https://www.mingw-w64.org/) which supports the GCC compiler on Windows systems. You can get this through the installation packages for [MSYS2](https://www.msys2.org/).
	- *For MacOS*: see this [link](https://macappstore.org/gcc/) to install GCC.
- The relevant Makefiles are included in the corresponding folders, so the compilation can be done by running the command line `make` in a terminal opened in the folder. 
- The compiled files should be in the same directory than the python scripts/notebooks in which the cycle matching code is invoked.

### Python

#### Python Dependencies
This repository uses a few packages beyond python built-ins, and most of these are only necessary for post-hoc analysis code in `visualization`. We recommend managing this package's environment and dependencies through [Anaconda/miniconda](https://www.anaconda.com/) and [conda-forge](https://conda-forge.org/). Dependencies are listed below and grouped by module. 

`utils_match`
- [numpy](https://numpy.org/)
	
`visualization`
- [POT (python optimal transport library)](https://pythonot.github.io/index.html)
- [OpenCV](https://pypi.org/project/opencv-python/) {optional alternative to POT as Wasserstein backend}
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [pandas](https://pandas.pydata.org/)
- [seaborn](https://seaborn.pydata.org/)
- [matplotlib](https://matplotlib.org/stable/index.html)


Note that the [OpenCV](https://pypi.org/project/opencv-python/) package is an optional alternative to [POT](https://pythonot.github.io/index.html) as the backend for Wasserstein computations; the code importing this package can be disabled without affecting overall functionality. In our problem regime, we found [POT](https://pythonot.github.io/index.html) to have performance advantages over [OpenCV](https://pypi.org/project/opencv-python/), hence its status as the default option.

## Academic use

This code is available and is fully adaptable for individual user customization. If you use these methods, please cite the following works:

```tex
@misc{garcia-redondo_fast_2022,
	title = {Fast {Topological} {Signal} {Identification} and {Persistent} {Cohomological} {Cycle} {Matching}},
	url = {http://arxiv.org/abs/2209.15446},
	urldate = {2022-10-03},
	publisher = {arXiv},
	author = {García-Redondo, Inés and Monod, Anthea and Song, Anna},
	month = sep,
	year = {2022},
	note = {arXiv:2209.15446 [math, stat]},
	keywords = {Mathematics - Algebraic Topology, Statistics - Machine Learning},
}
```

```tex
@misc{easley2023comparingrepresentationshighdimensionaldata,
      title={Comparing representations of high-dimensional data with persistent homology: a case study in neuroimaging}, 
      author={Ty Easley and Kevin Freese and Elizabeth Munch and Janine Bijsterbosch},
      year={2023},
      eprint={2306.13802},
      archivePrefix={arXiv},
      primaryClass={cs.CG},
      url={https://arxiv.org/abs/2306.13802}, 
}
```

## References
[1] Reani, Yohai, and Omer Bobrowski. 2021. ‘Cycle Registration in Persistent Homology with Applications in Topological Bootstrap’, January. https://arxiv.org/abs/2101.00698v1.

[2] Bauer, Ulrich. 2021. ‘Ripser: Efficient Computation of Vietoris-Rips Persistence Barcodes’. Journal of Applied and Computational Topology 5 (3): 391–423. https://doi.org/10.1007/s41468-021-00071-5.

[3] Bauer, Ulrich, and Maximilian Schmahl. 2022. ‘Efficient Computation of Image Persistence’. ArXiv:2201.04170 [Cs, Math], January. http://arxiv.org/abs/2201.04170.

[4] Čufar, Matija, and Žiga Virk. 2021. ‘Fast Computation of Persistent Homology Representatives with Involuted Persistent Homology’. ArXiv:2105.03629 [Math], May. http://arxiv.org/abs/2105.03629.

[5] T. Easley, K. Freese, E. Munch, and J. Bijsterbosch, “Comparing representations of high-dimensional data with persistent homology: a case study in neuroimaging,” Nov. 23, 2023, arXiv: arXiv:2306.13802. doi: [10.48550/arXiv.2306.13802](https://arxiv.org/abs/2306.13802).
