# Dynamics analysis with Koopman

Koopman operator analysis of pedestrian dynamics from trajectory data, built on
[datafold](https://datafold-dev.gitlab.io/datafold/).

This repository holds the code to reproduce the results of the clustering paper:
trajectories of 11 experiments from the
[Jülich pedestrian data archive](https://ped.fz-juelich.de/da/) are turned into
density and velocity fields, and a Koopman operator is fitted to each experiment.

## Associated publication

If you use this code, please cite the paper:

S. Kern, N. Bode, G. K&ouml;ster, "Bottlenecks are not created equal: A taxonomy of
pedestrian crowd dynamics", submitted to *Transportation Research Part C: Emerging Technologies*, 2026.

```bibtex
@unpublished{kern-2026,
  author = {Kern, S. and Bode, N. and K{\"o}ster, G.},
  title  = {Bottlenecks are not created equal: A taxonomy of pedestrian crowd dynamics},
  note   = {Submitted to Transportation Research Part C: Emerging Technologies},
  year   = {2026}
}
```

## Setup

Requirements: Python 3.10 (tested with 3.10.14) on macOS (arm64).

We manage the Python version with [pyenv](https://github.com/pyenv/pyenv) and the
virtual environment with
[pyenv-virtualenvwrapper](https://github.com/pyenv/pyenv-virtualenvwrapper).

```bash
git clone <this repository>
cd clustering-crowd-dynamics

# virtual environment setup
pyenv shell 3.10.14
mkvirtualenv clustering-crowd-dynamics

pip install -e ".[notebook]"

python patch/apply_patches.py -y
```

- **Install in editable mode** (`-e`). The package finds `configs/`, `data/` and
  `runs/` relative to the repository root.
- **Apply the patches.** `patch/apply_patches.py` replaces four files of the
  installed datafold 2.0.1 with symlinks to `patch/datafold/`. The patched files
  are required to reproduce the results. Use a dedicated virtual environment, since
  the patch changes the installed library.
- **Keep pandas at 2.2 or newer.** Older versions parse some trajectory
  coordinates differently, which changes the discretized fields. `pyproject.toml`
  pins the supported range.

## Repository layout

| Path                                  | Content                                                                                                          |
|---------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `src/dynamics_analysis_with_koopman/` | The package: config handling, data preprocessing, model fitting, plots.                                          |
| `configs/clustering/`                 | One preprocessing config (`*_dmd.yaml`) and one learning config (`*_takens_dmd_checkpoint.yaml`) per experiment. |
| `papers/clustering/`                  | Data conversion notebook, batch python script, cutoff windows, scenario overview and notebooks to create all figures used in the paper.                |
| `patch/`                              | Modified datafold files and the script that links them.                                                          |
| `data/`, `runs/`                      | Created locally, not tracked: trajectories and run outputs.                                                      |

## Reproducing the clustering paper

Reproduces the results of the paper: the Koopman models of the 11 Jülich experiments, split into 
25 scenarios, their clustering, and the figures. Set up the environment first, as described above.
Run every command from the repository root.

| File                                  | Purpose                                                                                                            |
|----------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `papers/clustering/run_overview.csv`                | One row per scenario: description, config, reference and the download link of the data. |
| `papers/clustering/load_juelich_trajectories.ipynb` | Converts the raw Jülich trajectories into the input format of the pipeline.               |
| `papers/clustering/batch_run_main.py`               | Runs the pipeline for all experiments in three steps.                                     |
| `papers/clustering/cutoff_settings.csv`             | The time window kept per trajectory file, applied as step 2 of the pipeline.              |
| `papers/clustering/compute_wasserstein_clusters.ipynb` | Clusters the scenarios by the distance between their Koopman spectra, see [step 4](#4-cluster-the-experiments). |
| Figure notebooks in `papers/clustering/`            | Turn the runs into the paper's figures, see [step 5](#5-create-the-figures).               |

### 1. Download the data

Download the trajectory archives from the links in the `perma link` column of
`papers/clustering/run_overview.csv`. The links point to the archive revisions used in the paper.
Unpack them into `data/juelich_db/`, so that the notebook finds these folders:

```text
data/juelich_db/
├── 2006bottleneck/2006bottleneck2_trajectories/
├── 2009corner/2009corner_open_trajectories_txt/
├── 2009tjunction/2009tjunction_trajectories_txt/
├── 2009unidirClosed/2009unidirectional_closed_trajectories_txt/
├── 2009unidirOpen/2009unidirectional_open_trajectories_txt/
├── 2013bidirectional/BI_CORR/BI_CORR_400_A/
├── 2013bidirectional/BI_CORR/BI_CORR_400_B/
├── 2013crossing90/2013crossing_90_trajectories_txt/
├── 2013crossing120/2013crossing_120_trajectories_txt/
├── 2013singleFile/PHAS_SEP/
├── 2013unidirectional/                      (trajectory files directly inside)
├── 2017sime/2017sime_trajectories_txt/
└── 2022wheelchair_luggage_bottleneck/2022wheelchair_luggage_bottleneck_trajectories_txt/
```

To keep the archives elsewhere, set the variable `JUELICH_DB_DIR` in `load_juelich_trajectories.ipynb` accordingly.

### 2. Convert the trajectories

Use the notebook `load_juelich_trajectories.ipynb` to shift, and
where needed to rotate, each experiment into the positive quadrant. It rescales
positions to meters and adds the simulation time from the corresponding frame rate. The result
lands in `data/juelich_trajectories/<scenario>/train/`, about 1.2 GB.

### 3. Run the pipeline

```bash
python papers/clustering/batch_run_main.py 1 2 3
```

| Step         | What it does                                                                                                                                                                      | Config per scenario                     |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|
| 1 preprocess | Reads the converted trajectories and computes density and velocity fields on a grid. Writes one time series per trajectory file, plus ground-truth plots and gifs. Fits no model. | `<scenario>_dmd.yaml`                   |
| 2 cutoff     | Trims each time series to the window in `cutoff_settings.csv`. Keeps the untrimmed copy in `<run>/original_ts/`.                                                                  | none                                      |
| 3 learn      | Fits an EDMD model on time-delay coordinates of the trimmed series of the matching step 1 run.                                                                                    | `<scenario>_takens_dmd_checkpoint.yaml` |

All three steps together take about 1.5 hours on an Apple silicon laptop.
Steps 2 and 3 reuse the output of step 1, so they can be repeated without step 1.
The cutoff windows were chosen by inspecting the publicly available videos as well as the gifs of step 1.

Useful options:

```bash
python papers/clustering/batch_run_main.py 3 --only 2006bottleneck 2009corner   # a subset of scenarios
python papers/clustering/batch_run_main.py --help                                # all options
```

Every run writes its outputs to `runs/clustering/<config name>_<timestamp>/`. A run folder contains:

- `config.yaml`, the fully resolved settings, and `metrics.yaml`, the run metrics
- `main.log`, the full log
- `data/`, the time series (step 1) or the eigenvalues, modes and predictions (step 3)
- `evaluation/`, `interpretation/` and `gifs/`, the plots

The steps hand run ids to each other through `papers/clustering/juelich_run_registry.json`.
It maps every scenario to its latest preprocessing run and learning run, with a
status for each. An scenario whose step failed is marked `failed` and skipped by
later steps. Each step prints the run ids it created when it finishes.

### 4. Cluster the scenarios

`compute_wasserstein_clusters.ipynb` produces the main results of the paper. It
filters the eigenvalues of each fitted model by their residuals and computes the
Wasserstein distance between the spectra of every pair of scenarios. It then
clusters the scenarios hierarchically and tests the clusters for significance.

It writes the distance matrix, the cluster projection and the dendrogram to `wasserstein_clusters/`. It also writes the scenario overview
table as `run_overview_portrait.tex` and `run_overview_references.bib`.

### 5. Create the figures

| Notebook | Output |
| --- | --- |
| `plot_ground_truth.ipynb` | Ground-truth density and velocity grids with the measurement cell, and the time series of the method overview, in `ground_truth/`. |
| `plot_window_reconstruction.ipynb` | Predicted and measured observables over the prediction window, and the reconstruction error table, in `window_reconstruction/`. |
| `eigenvalues.ipynb` | Eigenvalue analysis and plots for pair-wise review of the distance measures, in `eigenvalues/`. |
| `plot_modes_with_gifs.ipynb` | Koopman modes and mode animations, in `modes/`. |


### Running a single config

The batch script calls the package entry point once per config. You can do the same by hand:

```bash
python -m dynamics_analysis_with_koopman.main -s clustering/2006bottleneck_dmd.yaml
python -m dynamics_analysis_with_koopman.main --help
```

Settings are resolved in three layers. The defaults are defined in
`config/configs.py`, the YAML file overrides them, and command-line flags override
both. Each run stores its fully resolved settings in its own `config.yaml`.

A config with `data.load_data_from_checkpoint` reuses the preprocessed time series
of an earlier run instead of reading raw trajectories.

## Contact

For questions about the code or the paper, contact Sabrina Kern at
[sabrina.kern@hm.edu](mailto:sabrina.kern@hm.edu).

## Declaration of generative AI and AI-assisted technologies

During the preparation of this repository, the authors used GitHub Copilot and Claude Code in order to support the code creation, including re-factoring efforts. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content published in this repository. 
All conceptualization, text creation for the associated manuscript, as well as interpretations were developed independently by the authors.

## License

MIT, see [LICENSE](LICENSE).
The files in `patch/datafold/` are modified copies of [datafold](https://gitlab.com/datafold-dev/datafold),
which is also MIT licensed. Its notice is in [patch/datafold/LICENSE](patch/datafold/LICENSE).
