# ARC ((Vocabulary) Anchor-based Representation Clustering) for RAG

## Data

The datasets are attached in the repo, their official links are as follows:

- [HumanEval](https://huggingface.co/datasets/openai_humaneval)
- [NumpyEval](https://github.com/microsoft/PyCodeGPT/tree/main/cert/pandas-numpy-eval)
- [PandasEval](https://github.com/microsoft/PyCodeGPT/tree/main/cert/pandas-numpy-eval)

## Code

- `arc.py`: the entry point to run experiments
- `arc_cfg.py`: for dynamic configurations
- `arc_focus.py`: the pipeline and dependent functions
- `arc_layer.py`: the models and modules
- `arc_learner.py`: the trainer to train and test model
- `arc_load.py`: for loading data, benchmark, and metrics
- `arc_shared.py`: for static configurations
- `arc_tmp.py`: for running a summary on the results
- `arc_util.py`: the common utility functions

## Quick Search

- Code: check `src/*.py` files
- Scripts: check `src/*.sh` and `src/*.job` files
- Logs: check `logs/**/*.out` files
- Results: check `gens/**/*.jsonl` files

And also, you can check `data/*` and `docs/*` for data and model details

## How to reproduce

If you have access to a Slurm Cluster, do following steps:
1. install and activate the conda environment (check `src/_arc.sh`)
2. submit the tasks you want to run (check `src/*.job`)

If you want to run somewhere else, simply regard `*.job` files as `*.sh` to execute them

---
