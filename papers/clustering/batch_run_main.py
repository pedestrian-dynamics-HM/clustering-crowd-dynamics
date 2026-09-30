"""Step-wise batch driver for the clustering paper experiments.

Every config this drives lives in configs/clustering, so the package derives
meta.config_subdir='clustering' for it automatically and every run lands in
runs/clustering/<run_id> with no extra flags needed -- see RunMetaConfig.config_subdir.
The configs' own meta.scenario is left alone, holding each config's experiment
identity (2006bottleneck, 2013crossing90, ...): it is purely a data-loading
concern now, unrelated to where run artifacts land. What used to be called
"scenario" in this script is that same identity; it is called *experiment* here
so it is not confused with meta.scenario.

The workflow is three steps that have to run in order, with a manual inspection
of the cutoff settings in between:

    step 1 (preprocess)  run <experiment>_dmd.yaml for every experiment, with
                         --preprocess-only and --data-path (see
                         DEFAULT_RAW_DATA_DIR, data/juelich_trajectories by
                         default). These runs read the raw
                         trajectories, compute the density and velocity fields
                         for every time step, write one timeseries CSV per
                         series id into <run>/data and plot the ground truth --
                         but fit no model, since step 3 is what learns.
                         The run ids are recorded in the batch registry.

    step 2 (cutoff)      trim the timeseries of the step 1 runs to the relevant
                         window (as defined in the cutoff settings CSV) and
                         keep the untrimmed copies in <run>/original_ts.

    step 3 (learn)       run <experiment>_takens_dmd_checkpoint.yaml for every
                         experiment, each loading from the step 1 checkpoint of
                         the same experiment, i.e. the trimmed data.

Usage::

    python papers/clustering/batch_run_main.py 1
    python papers/clustering/batch_run_main.py 2
    python papers/clustering/batch_run_main.py 3
    python papers/clustering/batch_run_main.py 1 2 3     # all of it in one go
    python papers/clustering/batch_run_main.py 3 --only 2006bottleneck 2009corner

Steps communicate through the registry (``--registry``, by default
``papers/clustering/juelich_run_registry.json``): step 1 writes the experiment ->
run id mapping, steps 2 and 3 read it. That is what pairs
``2006bottleneck_takens_dmd_checkpoint`` with the run of ``2006bottleneck_dmd``.

The registry holds one entry per experiment, its most recent pair of runs. Step 1
replaces the entry of every experiment it preprocesses (dropping the learning run
of the previous pair, which belongs to the data it just replaced) and leaves the
other experiments alone, so the registry fills up across several partial runs.

main.py would assign a run id itself, but only once it starts -- and the id has
to be known here to record it. So every run gets its id up front, from the
package's own ``_generate_run_id``, and is started with ``--run-id``.
"""

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from dynamics_analysis_with_koopman.config.configs import DATA_DIR, PROJECT_ROOT, RUN_ARTIFACTS_DIR
from dynamics_analysis_with_koopman.config.loading import _generate_run_id

CONFIG_SUBDIR = 'clustering'
RUN_ARTIFACTS_SUBDIR = f'{RUN_ARTIFACTS_DIR}/{CONFIG_SUBDIR}'

# One entry per experiment. The config file names are derived from it, so an
# experiment is added or removed here once and applies to every step.
EXPERIMENTS = [
    '2006bottleneck',
    '2009corner',
    '2009tjunction',
    '2009unidirClosed',
    '2009unidirOpenBs',
    '2009unidirOpenC',
    '2013bidirectionalA',
    '2013bidirectionalB',
    '2013crossing90A',
    '2013crossing90B',
    '2013crossing90C',
    '2013crossing90D',
    '2013crossing90E',
    '2013crossing90F',
    '2013crossing90G',
    '2013crossing120A',
    '2013crossing120B',
    '2013crossing120C',
    '2013singleFile',
    '2013unidirectionalB',
    '2013unidirectionalC',
    '2017simeB',
    '2017simeC',
    '2022wheelchairB',
    '2022wheelchairU',
]

PREPROCESS_SUFFIX = '_dmd.yaml'
LEARNING_SUFFIX = '_takens_dmd_checkpoint.yaml'

# Lives next to this script rather than under runs/ .
# It serves as the index for the clustering and the paper visualizations.
DEFAULT_REGISTRY = Path(__file__).resolve().parent / 'juelich_run_registry.json'

# The script requires the pre-processed trajectories from the juelich data base.
# This is done by executing all cells in the load_juelich_trajectories.ipynb, already imposing the observation area
# The cut-offs were manually obtained by reviewing the videos and gifs, one line per time series.
DEFAULT_RAW_DATA_DIR = str(PROJECT_ROOT / DATA_DIR / 'juelich_trajectories')
DEFAULT_CUTOFF_SETTINGS = str(Path(__file__).resolve().parent / 'cutoff_settings.csv')

# Where step 2 keeps the untrimmed copies, inside each run's own folder.
ORIGINAL_TS_DIR = 'original_ts'
CUTOFF_COLUMNS = ['Experiment', 'Series ID', 'new start (t)', 'new end (t)']


def _resolve_run_dir(run_id: str) -> str | None:
    """Return a finished run's artifacts folder, relative to the project root.

    Every config here lives in configs/clustering, so meta.config_subdir resolves
    to 'clustering' automatically and the run lands in RUN_ARTIFACTS_SUBDIR -- the
    path is known rather than searched for; this only confirms it is really there.
    """
    run_dir = PROJECT_ROOT / RUN_ARTIFACTS_SUBDIR / run_id
    if not run_dir.is_dir():
        print(f'Warning: no run folder found at {run_dir}')
        return None
    return str(run_dir.relative_to(PROJECT_ROOT))


def _read_registry(registry_path: Path) -> dict:
    if not registry_path.exists():
        raise FileNotFoundError(
            f'No batch registry at {registry_path}. Run step 1 first, or point --registry '
            f'at the registry of the batch you want to continue.'
        )
    with open(registry_path) as f:
        return json.load(f)


def _write_registry(registry_path: Path, registry: dict) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2, sort_keys=True)
    print(f'Registry written to {registry_path}')


def _run_training(config_file: str, run_id: str, extra_args: list[str]) -> bool:
    """Run one training as a subprocess. Returns whether it succeeded.

    Failures are reported and swallowed: one broken experiment should not stop
    the remaining ones, the registry records which succeeded.
    """
    command = [
        sys.executable,
        '-m',
        'dynamics_analysis_with_koopman.main',
        '-s',
        f'{CONFIG_SUBDIR}/{config_file}',
        '--run-id',
        run_id,
        *extra_args,
    ]

    print('\n\n\n=====================================================================')
    start_time = datetime.datetime.now()
    print('Start execution at ', start_time)
    print(f'Running as run {run_id}: {" ".join(command)}')

    try:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,  # Captures stdout + stderr separately
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f'FAILED after {datetime.datetime.now() - start_time}: {" ".join(command)}')
        # The full log is in the run folder; the tail of stderr is enough to see
        # what went wrong without scrolling back through the whole batch.
        print('--- stderr (tail) ---')
        print('\n'.join((e.stderr or '').splitlines()[-20:]))
        print('=====================================================================')
        return False

    print(f'Done after {datetime.datetime.now() - start_time}')
    print('=====================================================================')
    return True


def step_preprocess(experiments: list[str], registry_path: Path, raw_data_dir: str) -> None:
    """Step 1: preprocess the raw trajectories for every experiment.

    The registry keeps one entry per experiment, holding its most recent pair of
    runs. Only the experiments preprocessed here are touched, so a run over part
    of the batch (--only) leaves the other experiments' entries in place.
    """
    # Reading the current registry allows to rerun a single scenario without removing the rest.
    registry = _read_registry(registry_path) if registry_path.exists() else {}
    registry.setdefault('runs', {})

    for experiment in experiments:
        config_file = f'{experiment}{PREPROCESS_SUFFIX}'
        run_id = _generate_run_id(config_file)
        # Execute the training in order to obtain the necessary time series data
        # based on the laboratory experiment trajectories.
        succeeded = _run_training(
            config_file,
            run_id,
            extra_args=[
                '--preprocess-only',  # this terminates the run before learning
                '--data-path',
                raw_data_dir,
            ],
        )

        # Replace the entry to always use the newest one for the downstream tasks.
        registry['runs'][experiment] = {
            'preprocess_config': config_file,
            'preprocess_run_id': run_id,
            'preprocess_run_dir': _resolve_run_dir(run_id) if succeeded else None,
            'preprocess_status': 'ok' if succeeded else 'failed',
        }
        registry['updated'] = datetime.datetime.now().isoformat(timespec='seconds')
        # Written after every experiment so an aborted batch keeps what it has.
        _write_registry(registry_path, registry)

    _report(registry, 'preprocess', experiments)


def _load_cutoff_settings(cutoff_settings_file: str) -> pd.DataFrame:
    """Read the cutoff windows. The first line of the CSV is a spacer, not the header."""
    cutoff_settings = pd.read_csv(cutoff_settings_file, delimiter=',', header=0)
    cutoff_settings = cutoff_settings[CUTOFF_COLUMNS].dropna(subset=CUTOFF_COLUMNS)
    cutoff_settings['Series ID'] = cutoff_settings['Series ID'].astype(int)
    return cutoff_settings


def _preprocess_runs_frame(registry: dict, experiments: list[str]) -> pd.DataFrame:
    """The step 1 runs that finished, as a frame to match cutoff rows against.

    'Experiment' is the batch key (the config file name), which is what the
    cutoff settings' 'Experiment' column is a prefix of -- not `meta.scenario`,
    which several configs share (2013crossing90A/B/C/F/G all set scenario
    2013crossing90).
    """
    preprocess_runs = pd.DataFrame(
        [
            {
                'Experiment': experiment,
                'runDir': PROJECT_ROOT / entry['preprocess_run_dir'],
            }
            for experiment, entry in sorted(registry.get('runs', {}).items())
            if entry.get('preprocess_status') == 'ok' and experiment in experiments
        ]
    )

    if len(preprocess_runs) == 0:
        raise ValueError('No successful preprocessing runs in the registry. Run step 1 first.')

    missing_dirs = preprocess_runs[~preprocess_runs['runDir'].apply(Path.exists)]
    if len(missing_dirs) > 0:
        raise FileNotFoundError(f'Run directories from the registry do not exist: {list(missing_dirs["runDir"])}')

    return preprocess_runs


def _find_timeseries(row: pd.Series, preprocess_runs: pd.DataFrame) -> tuple:
    """Locate the timeseries file for one cutoff row and the directory to read it from.

    A cutoff row applies to a run when the run holds a timeseries for that series
    id and the run's experiment key starts with the row's experiment name.
    """
    candidates = preprocess_runs[preprocess_runs['Experiment'].str.startswith(str(row['Experiment']))]

    file_name = f'timeseries_data_{int(row["Series ID"])}.csv'
    for _, run in candidates.iterrows():
        # Read the untrimmed data whenever a previous cut has stored it, so
        # applying the cutoff twice does not trim an already trimmed series.
        for data_dir in (ORIGINAL_TS_DIR, DATA_DIR):
            if (run['runDir'] / data_dir / file_name).exists():
                return run['Experiment'], run['runDir'], data_dir, file_name

    print(f'Did not find a timeseries for series {row["Series ID"]} of experiment {row["Experiment"]}')
    return None, None, None, None


def step_cutoff(experiments: list[str], registry_path: Path, cutoff_settings_file: str) -> None:
    """Step 2: cut the timeseries of the step 1 runs down to the relevant window.

    The untrimmed timeseries are kept in each run's `original_ts` directory, and
    read from there again on a repeat pass, so this is idempotent rather than
    cumulative. Restricted to `experiments` (--only), same as steps 1 and 3.
    """
    registry = _read_registry(registry_path)  # fail early if step 1 has not run
    print(f'Cutting the timeseries of the runs in {registry_path}')
    print(f'Using the cutoff settings in {cutoff_settings_file}')

    cutoff_settings = _load_cutoff_settings(cutoff_settings_file)
    # a row is in scope when it is a prefix of one of the selected experiment
    cutoff_settings = cutoff_settings[
        cutoff_settings['Experiment'].apply(
            lambda experiment: any(selected.startswith(experiment) for selected in experiments)
        )
    ]
    preprocess_runs = _preprocess_runs_frame(registry, experiments)
    print(f'Found {len(preprocess_runs)} preprocessing runs to cut')

    cutoff_settings[['Experiment', 'runDir', 'sourceDir', 'fileName']] = cutoff_settings.apply(
        _find_timeseries, axis=1, result_type='expand', args=(preprocess_runs,)
    )
    cutoff_settings = cutoff_settings[cutoff_settings['fileName'].notna()]
    print(f'{len(cutoff_settings)} timeseries to cut')

    for _, row in cutoff_settings.iterrows():
        # always reads the untrimmed data
        timeseries_data = pd.read_csv(row['runDir'] / row['sourceDir'] / row['fileName'])

        # filter the data
        timeseries_cut = timeseries_data[
            timeseries_data['timeStep'].between(int(row['new start (t)']), int(row['new end (t)']))
        ]

        # replace the timeseries with the cut, but keep the original in other folder
        (row['runDir'] / ORIGINAL_TS_DIR).mkdir(parents=True, exist_ok=True)
        timeseries_data.to_csv(row['runDir'] / ORIGINAL_TS_DIR / row['fileName'], index=False)
        timeseries_cut.to_csv(row['runDir'] / DATA_DIR / row['fileName'], index=False)

        print(
            f'{row["Experiment"]} / {row["fileName"]}: '
            f'{len(timeseries_data)} -> {len(timeseries_cut)} rows '
            f'(timesteps {int(row["new start (t)"])}-{int(row["new end (t)"])})'
        )

    print(f'\nCut {len(cutoff_settings)} timeseries in {cutoff_settings["Experiment"].nunique()} experiments')


def step_learn(experiments: list[str], registry_path: Path) -> None:
    """Step 3: learn on the trimmed timeseries, loading from the step 1 runs."""
    registry = _read_registry(registry_path)
    registry.setdefault('runs', {})

    for experiment in experiments:
        entry = registry['runs'].get(experiment)
        if entry is None:
            print(f'Skipping {experiment}: no step 1 run in the registry.')
            continue
        if entry.get('preprocess_status') != 'ok':
            print(f'Skipping {experiment}: its step 1 run did not succeed.')
            continue

        config_file = f'{experiment}{LEARNING_SUFFIX}'
        run_id = _generate_run_id(config_file)
        checkpoint_run_id = entry['preprocess_run_id']
        print(f'{experiment}: loading data from checkpoint {checkpoint_run_id}')
        succeeded = _run_training(
            config_file,
            run_id,
            extra_args=[
                '--load-data-from-checkpoint',
                checkpoint_run_id,
            ],
        )

        entry['learning_config'] = config_file
        entry['learning_run_id'] = run_id
        entry['learning_run_dir'] = _resolve_run_dir(run_id) if succeeded else None
        entry['learning_status'] = 'ok' if succeeded else 'failed'
        registry['updated'] = datetime.datetime.now().isoformat(timespec='seconds')
        _write_registry(registry_path, registry)

    _report(registry, 'learning', experiments)


def _report(registry: dict, stage: str, experiments: list[str]) -> None:
    """Print the run ids this invocation produced, for the evaluation notebooks.

    Restricted to the experiments just run: the registry also holds the entries
    of everything run before it, which this stage did not create.
    """
    entries = {experiment: entry for experiment, entry in registry.get('runs', {}).items() if experiment in experiments}
    succeeded = {
        experiment: entry[f'{stage}_run_id']
        for experiment, entry in entries.items()
        if entry.get(f'{stage}_status') == 'ok'
    }
    failed = [experiment for experiment, entry in entries.items() if entry.get(f'{stage}_status') == 'failed']

    print(f'\nCompleted the {stage} step, creating the following run_ids:')
    for experiment, run_id in succeeded.items():
        print(f'  {experiment}: {run_id}')
    print(f'\n{list(succeeded.values())}')
    if failed:
        print(f'\nFailed experiments: {failed}')


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'steps',
        nargs='+',
        choices=['1', '2', '3'],
        help='Steps to execute, in the given order: 1 preprocess, 2 cutoff, 3 learn.',
    )
    parser.add_argument(
        '--only',
        nargs='+',
        default=None,
        metavar='EXPERIMENT',
        help='Restrict the given steps to these experiments.',
    )
    parser.add_argument(
        '--registry',
        type=Path,
        default=DEFAULT_REGISTRY,
        help=f'Registry file the steps hand the run ids over with (default: {DEFAULT_REGISTRY}).',
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=DEFAULT_RAW_DATA_DIR,
        help=f'Directory holding the raw trajectories, used by step 1 (default: {DEFAULT_RAW_DATA_DIR}).',
    )
    parser.add_argument(
        '--cutoff-settings',
        type=str,
        default=DEFAULT_CUTOFF_SETTINGS,
        help='CSV defining the cutoff values, used by step 2.',
    )
    args = parser.parse_args(argv)

    if args.only is not None:
        unknown = [experiment for experiment in args.only if experiment not in EXPERIMENTS]
        if unknown:
            parser.error(f'Unknown experiments {unknown}. Known experiments: {EXPERIMENTS}')

    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    experiments = args.only if args.only is not None else EXPERIMENTS

    for step in args.steps:
        if step == '1':
            step_preprocess(experiments, args.registry, args.data_path)
        elif step == '2':
            step_cutoff(experiments, args.registry, args.cutoff_settings)
        elif step == '3':
            step_learn(experiments, args.registry)


if __name__ == '__main__':
    main()
