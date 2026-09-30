"""Module for feature generation and enrichment.

This module handles the generation of observable time series grids and
enrichment of data with external measurements.
"""

import datetime
import logging

import numpy as np
import pandas as pd

from ..config.configs import FullConfig
from .ingestion import MEASUREMENT_COLUMN_NAMES
from .grid import (
    _get_agent_count_grid,
    _get_agent_density_grid,
    _get_agent_velocity_grid,
    _get_obstacle_density,
    _get_obstacle_occupancy,
    _compute_convolved_velocity,
    _gaussian_kernel,
    DEFAULT_R,
)

log = logging.getLogger(__name__)


def _get_timestep_level_enrichment_data(input_file_name, enrich_timestep_data, tlim_start, data_files):
    """Read the enrichment data from the vadere output files and return a dictionary with the data per key.
    The data is averaged over all files and the time steps are shifted to start at 0.0s.
    The data is normalized by the maximum value over all time steps and measurement areas.
    """
    enrichment_counters = {}
    max_value_per_column_type = {}
    for series_id, path_list in data_files.items():
        enrichment_data_paths = [path.replace(input_file_name, enrich_timestep_data) for path in path_list]
        data = []
        for file in enrichment_data_paths:
            # read the txt as csv file
            data.append(pd.read_csv(file, sep=' ', header=0))
        df = pd.concat(data, ignore_index=True)

        # average the measurement values over all files
        column_names = df.columns
        for old_str, new_str in MEASUREMENT_COLUMN_NAMES.items():
            column_names = [f'{col.replace(old_str, "")}_{new_str}' if old_str in col else col for col in column_names]
        df.columns = column_names
        df = df.groupby(['timeStep']).mean().reset_index()

        # shift from timestep 1 to timestep 0 (Vadere just names the first timestep 1)
        df['timeStep'] = df['timeStep'] - 1

        # offset the time step by embedding timeshift, at this point the data is already shifted to start at 0
        if tlim_start is not None:
            df['timeStep'] = df['timeStep'] - tlim_start
        df = df[df['timeStep'] >= 0]

        # get the maximum value per column type
        for col_type in MEASUREMENT_COLUMN_NAMES.values():
            max_value_per_column_type[col_type] = max(
                max_value_per_column_type.get(col_type, 0), df.filter(like=col_type).max().max()
            )
        enrichment_counters[series_id] = df

    # normalize the data by the maximum value per column type
    for df in enrichment_counters.values():
        for col_type, max_value in max_value_per_column_type.items():
            if max_value > 0:
                df[df.filter(like=col_type).columns] = df.filter(like=col_type).div(max_value, axis=0)

    log.info(f'Enriched data collected from {enrich_timestep_data}.')

    return enrichment_counters


def generate_observable_timeseries_grid(
    config: FullConfig,
    data_dict,
    system_observables,
    num_runs,
    measurement_grid_size,
    grid_resolution,
    x_offset: float | None,
    y_offset: float | None,
):
    """
    The data_dict is an input with a single dataset per selection parameter. In case of average_across_simulations = True,
    the individual time entries are all concatenated into a single data frame and only aggregated here.
    We need the data to be at the level of each agent for weighted velocity calculations.
    This function returns a grid based data set, which is independent of the individual agents.
    """
    # want to create a single df per selection parameter with one column per system observable
    timeseries_data_dict: dict[int, pd.DataFrame] = {}
    header_cols = ['timeStep', 'x_idx', 'y_idx']

    for observable in system_observables:
        log.info(f'System observable specific processing for {observable=}...')
        feature_grid_data: dict[int, np.ndarray] = {}
        log.info(f'Get {observable} values from loaded data...')

        # iterate over the data dict and calculate the feature grid for each key
        # key for training and testing data is different due to prefix
        # each key may have a different length of timeseries if tlim[1] was None
        for key, df in data_dict.items():
            log.info(f'Processing {num_runs[key]} runs for selection parameter {key=}')
            ts_grid_shape = tuple([int(df['timeStep'].max()) + 1] + measurement_grid_size)

            # add pedestrian counter and average the velocity values for each grid cell
            # removes pedestrian id and will count across all simulations for averaging
            df['counter'] = 1
            if observable in ['density', 'count']:
                # TODO: could we instead use the mean here?
                agg_df = df.groupby(header_cols).agg(count=('counter', 'sum')).reset_index()

                # create a grid from data frame for each simulation run
                if observable == 'count':
                    log.info(f'Calculating agent count grid for {key=}.')
                    feature_grid_data[key] = (
                        _get_agent_count_grid(agg_df, ts_grid_shape) / num_runs[key]
                    )  # normalize by number of runs
                    log.info(f'Agent count grid calculated for {key=}.')

                else:  # must be density
                    log.info(f'Calculating agent density grid for {key=}.')
                    feature_grid_data[key] = (
                        _get_agent_density_grid(df=agg_df, ts_grid_shape=ts_grid_shape, grid_resolution=grid_resolution)
                        / num_runs[key]
                    )  # normalize by number of runs
                    log.info(f'Agent density calculated for {key=}.')

            elif observable in ['velocity_x', 'velocity_y']:
                agg_df = (
                    df.groupby(header_cols)
                    .agg(
                        velocity=(f'agent_{observable}', 'mean'),
                        count=('counter', 'mean'),  # mean for multiple runs
                    )
                    .reset_index()
                )

                feature_grid_data[key] = _get_agent_velocity_grid(agg_df[header_cols + ['velocity']], ts_grid_shape)

                if config.preprocessing.spatial_smoothing_neighborhood > 0:
                    # we use convolution to smooth the grid
                    count_grid = _get_agent_count_grid(
                        agg_df, ts_grid_shape
                    )  # no normalization, because we already have count mean
                    feature_grid_data[key] = _compute_convolved_velocity(
                        velocity_grid=feature_grid_data[key],
                        count_grid=count_grid,
                        grid_resolution=grid_resolution,
                        neighborhood_depth=config.preprocessing.spatial_smoothing_neighborhood,
                        kernel=_gaussian_kernel,
                        kernel_bandwidth=config.preprocessing.spatial_smoothing_bandwidth,
                    )

                log.info(f'Cell specific {observable} calculated for {key=}.')

            else:
                raise NotImplementedError(f'Check the choosen value for {observable=}')

        # add obstacle occupancy or density to the grid data (calcute it ONCE for all keys in dictionary)
        # TODO: make the obstacle grid run specific
        if config.preprocessing.obstacle_scaling > 0 and observable in ['density', 'count']:
            x_offset = 0 if x_offset is None else x_offset
            y_offset = 0 if y_offset is None else y_offset
            if observable == 'density':
                # get the density of the obstacles, same for all runs
                # TODO: make this run specific ??
                log.info('Calculating obstacle density...')
                start = datetime.datetime.now()
                obstacle_grid = config.preprocessing.obstacle_scaling * _get_obstacle_density(
                    measurement_grid_size,
                    config.preprocessing.obstacles,
                    x_offset,
                    y_offset,
                    grid_resolution,
                    R=DEFAULT_R,
                )
                log.info('Obstacle density calculated.')
                log.info(f'Time elapsed: {datetime.datetime.now() - start}')

            elif observable == 'count':
                obstacle_grid = config.preprocessing.obstacle_scaling * _get_obstacle_occupancy(
                    measurement_grid_size, config.preprocessing.obstacles, x_offset, y_offset, grid_resolution
                )

            # add the obstacle values to the grid density data for each series
            for key in feature_grid_data.keys():
                feature_grid_data[key] += obstacle_grid[None, :, :]  # broadcasting to all time steps

        # else:
        #     obstacle_grid = None

        # at this point, we have a feature grid for all keys and a fixed system observable
        # now, add the data to the timeseries data frame in the corresponding column
        log.info(f'Creating timeseries data for {observable=}...')
        for key in feature_grid_data.keys():
            log.debug(f'Creating timeseries data for {key=}')
            # create a new data frame from the 3d grids
            # https://stackoverflow.com/questions/46134827/how-to-recover-original-indices-for-a-flattened-numpy-array
            target_values = feature_grid_data[key]
            index = pd.MultiIndex.from_product(
                [range(s) for s in target_values.shape], names=['timeStep', 'y_idx', 'x_idx']
            )
            grid_data_dict = pd.DataFrame(
                {
                    observable: target_values.flatten(),  # concatenates the rows of the array
                },
                index=index,
            ).reset_index()
            # in case of multiple system observables, we need to merge the values into a single dataframe
            if timeseries_data_dict.get(key, None) is None:
                # dataframe is not yet created, so we can just assign the new data frame
                # to the corrsponding key in the dictionary
                timeseries_data_dict[key] = grid_data_dict
            else:
                # merge the data frame to add the new observable
                timeseries_data_dict[key] = pd.merge(
                    timeseries_data_dict[key],
                    grid_data_dict,
                    on=['timeStep', 'x_idx', 'y_idx'],  # , 'size'],
                    how='inner',
                )

        log.info(f'Timeseries data created for {observable=}.')
    return timeseries_data_dict
