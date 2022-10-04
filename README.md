# Persistent Cohomological Cycle Matching

This repository contains the code for the Persistent Cohomological Cycle Matching approach developed in the paper ["Fast Topological Signal Identification and Persistent Cohomological Cycle Matching" (García-Redondo, Monod, and Song 2022)](https://arxiv.org/abs/2209.15446).

## About C++

## About Jupyter

## Usage 

This repository is organised as follows.

- `applications`: folder with the applications included on Section 3 of the paper
	- `code`: same folder as before, needed to run the notebook of examples
	- `data`: folder featuring the dataset used for the examples on the paper. Note that some of the data we used required an institutional materials transfer agreement, so these data were not made available on this repository, and thus those examples are no reproducible.
	- `applications.ipynb`: notebook to reproduce all experiments in the paper. 
	- additional auxiliar scripts that are explained inside the notebook.
- `code`: folder containing the main scripts of code. You need to include this folder in the same directory as any python script that implements cohomological cycle matching.
	- `utils_data.py`: a python script with sampling functions on images, circles, and nii files for surfaces and volumes
	- `utils_PH.py`: a python script with functions to compute persistence, image-persistence and cycle matching
	- `utils_plot.py`: a python script for plotting functions
- `modified ripser`: folder containing the files of the modified versions of Ripser [1] and Ripser-image [2] needed to implement cycle matching. All the credit for the files in these folders should go to the authors in [1] and [2], the versions of the C++ code that we include here are exactly the same except for one line in the code, altered to extract the indices corresponding to the simplices of the persistence pairs after taking a lexicographic refinement.
	- `ripser-image-persistence-simple`: folder with the files for Ripser-image [2]. Go to the [original branch of the ripser repository](https://github.com/Ripser/ripser/tree/image-persistence-simple) for further detail.
	- `ripser-tight-representative-cycles`: folder with the files for Ripser [1] with an additional feature that computes representatives for the persistence bars in the barcode, as explained in [3]. Go to the [original branch of the ripser repository](https://github.com/Ripser/ripser/tree/tight-representative-cycles) for further detail.

## Academic use

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

### References
[1] Bauer, Ulrich. 2021. ‘Ripser: Efficient Computation of Vietoris-Rips Persistence Barcodes’. Journal of Applied and Computational Topology 5 (3): 391–423. https://doi.org/10.1007/s41468-021-00071-5.

[2] Bauer, Ulrich, and Maximilian Schmahl. 2022. ‘Efficient Computation of Image Persistence’. ArXiv:2201.04170 [Cs, Math], January. http://arxiv.org/abs/2201.04170.

[3] Čufar, Matija, and Žiga Virk. 2021. ‘Fast Computation of Persistent Homology Representatives with Involuted Persistent Homology’. ArXiv:2105.03629 [Math], May. http://arxiv.org/abs/2105.03629.
