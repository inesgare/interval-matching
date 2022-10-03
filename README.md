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

This repository is organised as follows.

- `code`: folder including the code we developed for the paper
	- `utils_data.py`: sampling functions on images, circles, and nii files for surfaces and volumes
	- `utils_PH.py`: functions to compute persistence, image-persistence and cycle matching
	- `utils_plot.py`: plot functions

- `examples`: applications included on Section 3 of the paper. Where possible, the data we used for those examples are also provided so that all experiments are fully reproducible. Note that some of the data we used required an institutional materials transfer agreement, so these data were not made available on this repository, and thus those examples are no reproducible.
