# Interval matching & related utilities

Refactored interval-matching code; mostly retains the logic and core implementation of [interval-matching](https://github.com/inesgare/interval-matching), but reorganizes and replaces some of it. The submodules in the files below contain `if name=="__main__":` contingencies that also allow them to be called as scripts from the command line; note that this use case will break any relative imports.

- `compute.py`: Jaccard index and (currently deprecated) Ripser calls (included to retain Python Ripser wrappers in absence of SLURM access)
- `collate_tagged_data.py`: aggregates cycle-matching data over ASCII bit-encoded tags into compact dictionary structure
- `create_ldm_images.py`: formats and write out Ripser inputs. Generalizes *bootstrap* cycle-matching to arbitrary precomputed metrics by building all Ripser-image inputs from the original metric input; only valid in bootstrapping case (see function `_subsamp_dZ` in this submodule).
- `extract.py`: collect and reformat persistence data from Ripser output
- `generate_subindex.py`: generate and bit-encode bootstrapping indices
- `matching.py`: perform cycle-matching operations
- `prevalence.py`: define and read/write prevalence scoring
