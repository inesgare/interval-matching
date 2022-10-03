# README - Persistent Cohomological Cycle Matching

This repository contains the code for the Persistent Cohomological Cycle Matching approach developed in the paper ["Fast Topological Signal Identification and Persistent Cohomological Cycle Matching" (García-Redondo, Monod, and Song 2022)](https://arxiv.org/abs/2209.15446).

This code is available and is fully adaptable for individual user customization. If you use the our methods, please cite as the following:

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

## Usage 

In this repository there is a folder named `code` that contains the following:

- `utils_data.py`: a python script with sampling functions on images, circles, and nii files for surfaces and volumes
- `utils_PH.py`: a python script with functions to compute persistence, image-persistence and cycle matching
- `utils_plot.py`: a python script for plotting functions
- `examples`: folder with the applications included on Section 3 of the paper. It is organised as follows:
	- `data`: folder featuring the dataset used for the examples on the paper. Note that some of the data we used required an institutional materials transfer agreement, so these data were not made available on this repository, and thus those examples are no reproducible.
	- `examples.ipynb`: notebook to reproduce all experiments in the paper. 
	- additional auxiliar scripts that are explained inside the notebook.
