"""Module for file loading and trajectory parsing.

This module handles reading trajectory data from files (both Vadere simulations
and experimental data) and converting them into DataFrames for further processing.
"""

import glob
import logging
import os
import re
from pathlib import Path

import pandas as pd

from ..config.configs import FullConfig

log = logging.getLogger(__name__)

# Constants for measurement column naming
MEASUREMENT_COLUMN_NAMES = {  # use the v prefix, so it is not a duplicate to calculated values
    'areaSpeed-PID': 'vspeed',
    'areaDensityCounting-PID': 'vnumPeds',  # can not contain "count"
    'crossings-PID': 'vsumCrossings',  # can not contain "count"
}


def _read_data_from_files(
    path: str,
    input_file_name: str,
    selection_parameters: list | None = None,
    averaging: bool = False,
    use_train_test_split: bool = True,
) -> tuple[dict[int, pd.DataFrame], dict[int, list[str]], dict[int, list[str]]]:
    """
    Read the data from the files in the given path and return a dictionary with the data per key.
    To use a train-test-split, the data needs to be stored in subdirectories "train" and "test".
    Otherwise, all data is considered training data.
    In case, the data is stored in subdirectories, we assume that the data comes from Vadere.
    In case the data is stored in the main directory, we assume that the data comes from the experiment and contains plain trajectories.
    """  # create the data dictionary here, to allow extending it when reading the data from train and test directories
    data_dict: dict[int, pd.DataFrame] = {}
    if 'train' not in os.listdir(path):
        raise FileNotFoundError(
            f'No training data was found in {path!r}. Training data needs to be stored in a '
            f"subdirectory 'train' of the scenario directory."
        )
    train_files = [
        f for f in os.listdir(os.path.join(path, 'train')) if '.txt' in f or '.csv' in f
    ]  # allow both csv and txt files

    # load the training data first, it needs to exist
    if len(train_files) == 0:
        # we are in a directory containing vadere subdirectories
        log.info('Load training data from vadere subdirectories.')
        data_dict, train_files_per_key = _get_vadere_data(
            os.path.join(path, 'train'),
            input_file_name=input_file_name,
            selection_parameters=selection_parameters,
            data_dict=data_dict,
            averaging=averaging,
        )
    else:
        # we are in a directory experiment data
        log.info('Load training data from experiment data files.')
        data_dict, train_files_per_key = _get_experiment_data(
            os.path.join(path, 'train'),
            data_dict=data_dict,
            selection_parameters=selection_parameters,
            averaging=averaging,
        )

    if len(train_files_per_key) == 0:
        raise FileNotFoundError(
            f'Did not find any training data in {path!r}. Please check the path, the scenario and the '
            f'selection parameter values.'
        )

    # check if we have testing data
    if 'test' in os.listdir(path):
        test_files = [f for f in os.listdir(os.path.join(path, 'test')) if '.txt' in f or '.csv' in f]  # may be empty
        if len(test_files) == 0:
            # we are in a directory containing vadere subdirectories
            log.info('Load test data from vadere subdirectories.')
            data_dict, test_files_per_key = _get_vadere_data(
                os.path.join(path, 'test'),
                input_file_name=input_file_name,
                selection_parameters=selection_parameters,
                data_dict=data_dict,
                averaging=averaging,
            )
        else:
            # we are in a directory experiment data
            log.info('Load test data from experiment data files.')
            data_dict, test_files_per_key = _get_experiment_data(
                os.path.join(path, 'test'),
                data_dict=data_dict,
                selection_parameters=selection_parameters,
                averaging=averaging,
            )
            if averaging:
                log.warning(
                    'Averaging is not implemented for experiment data. Please check if this was turned on on purpose.'
                )

    else:  # no test data found
        log.info('No test data found. No test data will be loaded.')
        test_files_per_key = {}

    # in case the test-train-split should not be considered, we make all selected data training data
    if not use_train_test_split:
        log.info(f'Due to {use_train_test_split=}, all loaded data is considered training data.')
        train_files_per_key.update(test_files_per_key)  # merges the dictionaries
        test_files_per_key = {}

    log.info(f'Timeseries data collected for {data_dict.keys()=}')

    return data_dict, train_files_per_key, test_files_per_key


def _group_trajectory_file_paths_by_key(
    path: str,
    selection_parameters: list | None = None,
    input_file_name: str | None = None,
    averaging: bool = False,
) -> dict[int, list[str]]:
    """
    Group trajectory file paths by key based on selection parameters and averaging settings.
    """
    filenames: dict[int, list[str]] = {}
    key_prefix = '200' if 'test' in path.split('/')[-1] else '100'

    # get the files in the directory
    if input_file_name is not None:
        # use the input file name to find the files (for vadere data)
        trajectory_file_paths = glob.glob(
            f'**/{input_file_name}', root_dir=path, recursive=True
        )  # assumes unique run id
    else:
        # use all trajectory files in the directory (for experiment data)
        trajectory_file_paths = [f for f in os.listdir(path) if f.endswith('.txt') or f.endswith('.csv')]

    if selection_parameters is None:  # we want all the files in the directory
        for file_path in trajectory_file_paths:
            if averaging:
                # want to create one timeseries from all files
                key = key_prefix
            else:
                # keep the files separate using their file names as id
                file_name = '_'.join(file_path.split('/')[-2:])  # needs to include the subdirectory for vadere data
                numbers = [str(v) for v in re.findall(r'\d+', file_name)]
                key = key_prefix + ''.join(numbers[:3] + [numbers[-1]])  # only keep the numbers, maximum of 5
            filenames.setdefault(int(key), []).append(os.path.join(path, file_path))

    else:  # selection_parameters is not None:
        # keep only the files that contain any selection parameter
        # create one time series per selection parameter
        for file_path in trajectory_file_paths:
            for key in selection_parameters:
                if (
                    f'_{key}_' in file_path or key == file_path.split('/')[0]
                ):  # allow the selection of unique simulations via full file name
                    if averaging:
                        numbers = [str(v) for v in re.findall(r'\d+', key)]
                    else:
                        file_name = '_'.join(
                            file_path.split('/')[-2:]
                        )  # needs to include the subdirectory for vadere data
                        numbers = [str(v) for v in re.findall(r'\d+', file_name)]
                    # create a key based on the selection parameter and the file name
                    key = key_prefix + ''.join(numbers[:3] + [numbers[-1]])
                    filenames.setdefault(int(key), []).append(os.path.join(path, file_path))
                    break  # do not check all other keys

    return filenames


def _get_vadere_data(
    path: str,
    input_file_name: str,
    selection_parameters: list | None,
    data_dict: dict[int, pd.DataFrame],
    averaging: bool = False,
) -> tuple[dict[int, pd.DataFrame], dict[int, list[str]]]:
    """
    Read the data from the files in the given path and return a dictionary with the data per key.
    In case of averaging, the data is concatenated into one data frame per key.
    NOTE: we want integer keys, because this corresponds with the TSCDataFrame later on
    """
    filenames = _group_trajectory_file_paths_by_key(
        path=path,
        input_file_name=input_file_name,
        selection_parameters=selection_parameters,
        averaging=averaging,
    )

    log.info(f'Found {len(filenames)} timeseries in {path=}')
    log.debug(f'Found {filenames.keys()} keys in {path=}')

    # per key, create a separate data frame containing all the corresponding file contents
    # in case of averaging, we concatenate all the files into one data frame
    # the true average is calculated later on during reduction of the df data
    data_dict = _vadere_trajectory_to_df(filenames, data_dict)

    log.info(f'Loaded data for {len(data_dict.keys())} keys.')
    log.debug(f'Loaded data length per key: { {key: len(data_dict[key]) for key in data_dict.keys()} }')

    return data_dict, filenames


def _vadere_trajectory_to_df(filenames, data_dict: dict[int, pd.DataFrame] | None = None):
    """
    Convert Vadere trajectory files to DataFrames.
    """
    if data_dict is None:
        data_dict = {}

    for key, files in filenames.items():
        data = []
        for i, f in enumerate(files):
            data.append(pd.read_csv(f, sep=' ', header=0))
            # append "i" to the front of the pedestrian ID (uniqueness for the aggregation and interpolation later on)
            data[-1]['pedestrianId'] = data[-1]['pedestrianId'].apply(lambda x: f'{hex(i)}{x}')  # noqa: B023
        data_dict[key] = pd.concat(data, ignore_index=True)
        # rename the columns to the correct names
        data_dict[key].columns = [col.split('-')[0] for col in data_dict[key].columns]
        data_dict[key].drop(columns=['targetId'], inplace=True)

    return data_dict


def _get_experiment_data(
    path: str,
    data_dict: dict[int, pd.DataFrame],
    selection_parameters: list | None,
    averaging: bool = False,
) -> tuple[dict[int, pd.DataFrame], dict[int, list[str]]]:
    """
    Read the data from the files in the given path and return a dictionary with the data per key.
    We assume a single timeseries per file, no averaging is done for this data.
    """
    # key is simply the enumeration of the file name (since it will not have any specific format)
    filenames = _group_trajectory_file_paths_by_key(
        path=path,
        selection_parameters=selection_parameters,
        averaging=averaging,
    )

    # per key, create a separate data frame containing all the corresponding file contents
    # NOTE: key is integer
    for key, files in filenames.items():
        data_dict[key] = pd.concat([pd.read_csv(f, sep=' ', header=0) for f in files], ignore_index=True)
        # check if at least all the required columns are present
        if not all(
            col in data_dict[key].columns
            for col in ['pedestrianId', 'simTime', 'endTime', 'startX', 'startY', 'endX', 'endY']
        ):
            raise ValueError(
                f'Data does not have the correct columns: {data_dict[key].columns}. Header has contain all of the following columns: pedestrianId simTime endTime startX startY endX endY'
            )

    return data_dict, filenames


def load_trajectory_data_to_grid(
    config: FullConfig,
) -> tuple[dict[int, pd.DataFrame], dict[int, list[str]], list[int], dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    """Load raw trajectories and convert to grid-based observables.

    Args:
        config: FullConfig with data loading and preprocessing settings.

    Returns:
        Tuple of (timeseries_data_dict, data_files, measurement_grid_size, enrichment_counters,
        position_data_dict). position_data_dict is the per-series, grid-reduced (but not yet
        aggregated into observables) trajectory data -- callers that want it persisted (see
        preprocessing.store_position_ts) are responsible for writing it themselves.
    """
    from .discretization import _select_time_steps, _get_time_discrete_trajectories_df
    from .grid import _reduce_grid_df
    from .observables import _get_timestep_level_enrichment_data, generate_observable_timeseries_grid

    scenario = config.meta.scenario
    log.info('Looking for files with %s in path %s', scenario, config.meta.raw_data_path)

    try:
        positions_data_dict, train_files, test_files = _read_data_from_files(
            path=str(Path(config.meta.raw_data_path) / scenario),
            input_file_name=config.data.input_file_name,
            selection_parameters=config.data.selection_parameters,
            averaging=config.preprocessing.average_across_simulations,
            use_train_test_split=config.data.use_train_test_split,
        )

        positions_data_dict = _select_time_steps(
            positions_data_dict,
            min_timestep=config.preprocessing.tlim[0],
            max_timestep=config.preprocessing.tlim[1],
            temporal_resolution=config.preprocessing.temporal_resolution,
        )

        positions_data_dict = _get_time_discrete_trajectories_df(
            positions_data_dict,
            temporal_resolution=config.preprocessing.temporal_resolution,
            interpolate=config.preprocessing.interpolate,
        )

    except FileNotFoundError as e:
        log.error('No trajectory data found at %s', Path(config.meta.raw_data_path) / scenario)
        raise e

    # Internal invariant: check_config_consistency normalises a scalar grid_resolution
    # into a two-element list before any data is loaded.
    assert isinstance(config.preprocessing.grid_resolution, list), (
        'Grid resolution must be a list of two floats for x and y direction.'
    )
    positions_data_dict, measurement_grid_size = _reduce_grid_df(
        data_dict=positions_data_dict,
        xlim=config.preprocessing.xlim,
        ylim=config.preprocessing.ylim,
        tlim=config.preprocessing.tlim,
        grid_resolution=config.preprocessing.grid_resolution,
    )

    data_files = train_files | test_files
    num_runs = {key: len(values) for key, values in data_files.items()}
    timeseries_data_dict = generate_observable_timeseries_grid(
        config=config,
        data_dict=positions_data_dict,
        system_observables=config.preprocessing.system_observables,
        num_runs=num_runs,
        measurement_grid_size=measurement_grid_size,
        grid_resolution=config.preprocessing.grid_resolution,
        x_offset=config.preprocessing.xlim[0],
        y_offset=config.preprocessing.ylim[0],
    )

    if config.data.enrich_timestep_data is not None:
        log.warning('Data is enriched with additional measurement data. This only works for Vadere runs at the moment.')
        enrichment_counters = _get_timestep_level_enrichment_data(
            data_files=data_files,
            input_file_name=config.data.input_file_name,
            enrich_timestep_data=config.data.enrich_timestep_data,
            tlim_start=config.preprocessing.tlim[0],
        )
    else:
        enrichment_counters = {}

    return timeseries_data_dict, data_files, measurement_grid_size, enrichment_counters, positions_data_dict
