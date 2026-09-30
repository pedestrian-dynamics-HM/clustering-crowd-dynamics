"""Module for temporal alignment and discretization.

This module handles converting continuous trajectory data into discrete time steps
through interpolation or expansion, and selecting time windows for processing.
"""

import logging
from functools import partial

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Constants for temporal resolution
DEFAULT_TEMP_RESOLUTION = 0.2

# Tolerance for converting a continuous time to a discrete grid step.
_GRID_SNAP_EPSILON = 1e-6


def _expand_row(time_step_size: float, row: pd.Series) -> pd.Series:
    """
    The (continuous) agent's step is expanded into a list of values for each discrete time step.
    Hence, the x, y and timeStep columns are replaced with a lists of values.
    """
    # NOTE: the time 0.0 is t = 1, time 0.2 is t = 2, etc.
    first_count_step = np.ceil(row['simTime'] / time_step_size - _GRID_SNAP_EPSILON)
    # need to make sure we start at the correct time step (want to shift forward)
    row['timeStep'] = np.arange(first_count_step, row['endStep'] + 0.1)

    # no interpolation, be just repeat the end value for the agent step
    row['x'] = [row['endX'] for _ in row['timeStep']]
    row['y'] = [row['endY'] for _ in row['timeStep']]

    # if the step is aligned, we need to use the start position in the first entry instead
    shift_from_frames = row['simTime'] % time_step_size
    if shift_from_frames == 0:
        row['x'][0] = row['startX']
        row['y'][0] = row['startY']

    return row


def _interpolate_positions(time_step_size: float, row: pd.Series) -> pd.Series:
    """
    Interpolates the positions of the agent for each discrete time step.
    This is needed if trajectory data has steps larger than the requested temporal resolution.
    Note that the offset for the first step is needed to fall in the rhythm of the discretization.
    """
    # interpolate the x and y values for each time step
    # need to make sure we start at the correct time step (want to shift forward)
    first_snapshot_t = np.ceil(row['simTime'] / time_step_size)  # we have t = 0 at 0.0s
    last_snapshot_t = np.floor(row['endTime'] / time_step_size)  # this is the last snapshot within the step

    # discretization creates relative shifts for timestep with length of time_step_size
    row['timeStep'] = np.arange(first_snapshot_t, last_snapshot_t + 1)
    snapshot_relative_times = row['timeStep'] * time_step_size - row['simTime']

    step_duration = row['endTime'] - row['simTime']
    row['x'] = [
        round(row['startX'] + s * (row['endX'] - row['startX']) / step_duration, 5) for s in snapshot_relative_times
    ]
    row['y'] = [
        round(row['startY'] + s * (row['endY'] - row['startY']) / step_duration, 5) for s in snapshot_relative_times
    ]

    return row


def _select_time_steps(
    data_dict: dict[int, pd.DataFrame],
    min_timestep: int | None = None,
    max_timestep: int | None = None,
    temporal_resolution: float = DEFAULT_TEMP_RESOLUTION,
) -> dict[int, pd.DataFrame]:
    """
    Filter the data based on the given time limits to reduce the comuptational efforts.
    It is okay to do coarse filtering here, because the data is further reduced after discretization later.
    NOTE: This function assumes that during processing, the time measurements were already shifted to start at 0.0s.
    """
    log.debug(f'Removing all time steps larger than {max_timestep=} in order to reduce discretization efforts.')
    result = dict(data_dict)

    if min_timestep is not None:
        for key, run_data in result.items():
            # need a small buffer of 5s to ensure to include the start of the actual pedestrian's step
            result[key] = run_data[(run_data['simTime'] >= min_timestep * temporal_resolution - 5)]

    if max_timestep is not None:
        for key, run_data in result.items():
            # need a small buffer of 5s to make sure we include all pedestrians until their last step
            result[key] = run_data[(run_data['endTime'] <= max_timestep * temporal_resolution + 5)]

    return result


def _get_time_discrete_trajectories_df(
    traj_data_dict: dict[int, pd.DataFrame], temporal_resolution: float, interpolate: bool = False
) -> dict[int, pd.DataFrame]:
    """
    Transform trajectory dicts in a format with single entry per timestep per grid cell
    and average across the velocities in each grid cell.
    """
    # header as defined in loading function
    # Vadere header: pedestrianId simTime endTime startX startY endX endY
    # trajectory header for experiments: id timeStep x y
    log.info(f'Discretizing the trajectory data with {temporal_resolution=}')

    result: dict[int, pd.DataFrame] = {}
    for key, sim_data in traj_data_dict.items():
        log.info(f'Discretizing the trajectory data for {key=}')
        # create a timestep column, first timestep is 1 for easier calculation, shift the time step by 1 later
        sim_data['startStep'] = (sim_data['simTime'] / temporal_resolution).astype(int) + 1
        sim_data['endStep'] = (sim_data['endTime'] / temporal_resolution).astype(int) + 1

        # Add additional rows for each entry in the simulation
        # one row should represents the movement of the agent in one *discrete* time step
        log.info(f'Expanding the trajectory data to match temporal resolution of {temporal_resolution=}')

        # check if the the granularity in the data is larger or lower than the temporal resolution
        # i.e. if the startStep and endStep are the same, we have multiple entries for the same time step
        if len(sim_data.loc[sim_data['startStep'] == sim_data['endStep']]) == 0:
            # create a new column with velocity into x and y direction
            # TODO: check if this makes sense to calculate prior to discretization and then just keep the values
            t_delta = sim_data['endTime'] - sim_data['simTime']  # this is a vector !
            # we may have read them via the experimental data input
            if 'agent_velocity_x' not in sim_data.columns and 'agent_velocity_y' not in sim_data.columns:
                sim_data['agent_velocity_x'] = (sim_data['endX'] - sim_data['startX']) / t_delta  #
                sim_data['agent_velocity_y'] = (sim_data['endY'] - sim_data['startY']) / t_delta  #
                log.info(f'Calculated the agent velocity for {key=}')

            # i.e. all the steps are larger than the temporal resolution
            log.info(f'Set {temporal_resolution=} is higher than in the trajectory data. Interpolating the steps...')
            if interpolate:
                sim_data = sim_data.apply(
                    partial(_interpolate_positions, temporal_resolution), axis=1
                )  # interpolates the steps
            else:
                sim_data = sim_data.apply(
                    partial(_expand_row, temporal_resolution), axis=1
                )  # adds a list of values per time step
            sim_data = sim_data.explode(['x', 'y', 'timeStep']).reset_index(drop=True)  # explodes the list of values
            log.info(f'Expanded the trajectory data for {key=}')

            # get all the duplicates on pedestrian id and timestep
            # should only ever happen in interpolation
            duplicates = sim_data[sim_data.duplicated(subset=['pedestrianId', 'timeStep'], keep=False)]
            if len(duplicates) > 0:
                log.warning(
                    f'Found {len(duplicates)} duplicate entries in the trajectory data due to numerical inaccuracies.'
                )
                log.warning('Removing these duplicates...')
                sim_data = sim_data.drop_duplicates(subset=['pedestrianId', 'timeStep'], keep='first')

        else:
            # we have more snapshots than we need, hence we merge multiple entries into one row per time step
            log.info('No need to expand the trajectory data. Discretizing the data by merging steps...')

            sim_data['timeStep'] = sim_data['startStep']
            sim_data['x'] = sim_data['startX']
            sim_data['y'] = sim_data['startY']

            # get the entries where endStep is equal to startStep
            # TODO: check if this works with averaging!
            if 'agent_velocity_x' in sim_data.columns and 'agent_velocity_y' in sim_data.columns:
                # we may have read them via the experimental data input
                sim_data.sort_values(by=['simTime'], inplace=True)

                sim_data = (
                    sim_data.groupby(['pedestrianId', 'timeStep'])
                    .agg(
                        {
                            'startStep': 'first',  # Take the first value for start position
                            'endStep': 'last',  # Take the last value for end position
                            'x': 'first',  # Take the first value for start position
                            'y': 'first',  # Take the first value for start position
                            'endX': 'last',  # Take the last value for end position
                            'endY': 'last',  # Take the last value for end position
                            'agent_velocity_x': 'first',  # Take the mean value for velocity
                            'agent_velocity_y': 'first',  # Take the mean value for velocity
                        }
                    )
                    .reset_index()
                )

            else:
                sim_data.sort_values(by=['simTime'], inplace=True)
                sim_data = (
                    sim_data.groupby(['pedestrianId', 'timeStep'])
                    .agg(
                        {
                            'startStep': 'first',  # Take the first value for start position
                            'endStep': 'last',  # Take the last value for end position
                            'x': 'first',  # Take the first value for start position
                            'y': 'first',  # Take the first value for start position
                            'endX': 'last',  # Take the last value for end position
                            'endY': 'last',  # Take the last value for end position
                        }
                    )
                    .reset_index()
                )

                # get the velocity now that we have the correct resolution
                sim_data['agent_velocity_x'] = (sim_data['endX'] - sim_data['x']) / temporal_resolution
                sim_data['agent_velocity_y'] = (sim_data['endY'] - sim_data['y']) / temporal_resolution

        # we started at 0.0 seconds with the timestep 1 to make calculations easier
        # however, it is easier to think about 0.0s seconds as t = 0, so we shift by 1
        sim_data['timeStep'] = sim_data['timeStep'] - 1

        # drop columns that are not in header columns we want to keep
        result[key] = sim_data[['timeStep', 'x', 'y', 'agent_velocity_x', 'agent_velocity_y']].copy()

    return result
