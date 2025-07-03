# Visualization and post-hoc analysis

Modules for post-hoc analysis of persistence diagrams composed of matched intervals. Implements variants of the Wasserstein distance (see `diagram_distances.py`) and visualizations of distributions of persistence features. The `toy_models.py` submodule implements several reference spaces for benchmark and analysis.

- `bootstrap_dists.py`: computes distance between diagrams each containing bootstrap-matched cycles
- `compare_topostats.py`: compares distributions of topological features
- `comp_permtest_dists.py`: compute distribution of distances between data and its null-permuted counterparts
- `diagram_distances.py `: define and compute distance metrics (i.e., Wasserstein variants) between persitence diagrams
- `distributional_summaries.py`: create graphical summaries of topological features
- `prevwt_PD.py`: prevalence-weighted persistence diagram
- `toy_models.py`: submodule for creation of several toy-model validation spaces (concentric circles, S<sup>1</sup> wegde S<sup>2</sup> wedge S<sup>1</sup>, S<sup>2</sup> with a diameter, and the torus T<sup>2</sup>)
