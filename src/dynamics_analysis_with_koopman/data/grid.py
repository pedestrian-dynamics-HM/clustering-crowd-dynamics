"""Module for spatial discretization and kernel operations.

This module handles grid-based operations including obstacle occupancy/density,
agent density calculation, velocity grid computation, and kernel-based smoothing.
"""

import logging

import numpy as np
import pandas as pd
from numba import jit
from scipy import integrate
from scipy.ndimage import convolve

log = logging.getLogger(__name__)

# Constants for spatial resolution
DEFAULT_R = 0.7
DEFAULT_GRID_RESOLUTION = [0.2, 0.2]


# use of jit improves performance by 2x
@jit
def _gauss_density(x: np.ndarray, z: np.ndarray, R: float = DEFAULT_R) -> np.ndarray:
    """
    Computes the kernel density for a single pedestrian.
    """
    distances = np.linalg.norm(x - z) ** 2
    R_sq = 2 * R**2
    coeff = 1 / (np.pi * R_sq)
    gauss_values = coeff * np.exp(-distances / R_sq)

    return gauss_values


def _get_occupied_areas(obstacles: list, x_offset: float, y_offset: float) -> list[np.ndarray]:
    """
    Transform the list of obsatacles with (x,y) coordinates of the lower right corner,
    height and width into the format [i_min, i_max, j_min, j_max] in cell indices.

    The meters are transformed into cell indices using the grid resolution.
    """

    occupied_areas = []
    for obstacle in obstacles:
        x_min = obstacle[0][0] - x_offset
        x_max = x_min + obstacle[1]  # add the width of the obstacle

        y_min = obstacle[0][1] - y_offset
        y_max = y_min + obstacle[2]  # add the height of the obstacle

        # x corresponds with columns, y with rows
        i_min, i_max = y_min, y_max
        j_min, j_max = x_min, x_max

        occupied_areas.append(np.array([i_min, i_max, j_min, j_max]))
    return occupied_areas


def _get_obstacle_occupancy(
    measurement_grid_size: list[int],
    obstacles: list,
    x_offset: float,
    y_offset: float,
    grid_resolution: list[float] = DEFAULT_GRID_RESOLUTION,
) -> np.ndarray:
    """
    Transfers the obstacle information into a grid with the same size as the measurement grid
    The cells with obstacles are set to 2.
    """
    obstacle_count_grid = np.full(measurement_grid_size, 0.0)

    occupied_areas = _get_occupied_areas(obstacles, x_offset, y_offset)

    # handle the resolution
    resolution_x, resolution_y = grid_resolution[0], grid_resolution[1]

    # apply the gaussian kernel for each obstacle
    for i_min, i_max, j_min, j_max in occupied_areas:
        # avoid numerical effects by adding half of the grid resolution
        i_min_idx, i_max_idx = int(i_min / resolution_y + 0.5), int(i_max / resolution_y + 0.5)
        j_min_idx, j_max_idx = int(j_min / resolution_x + 0.5), int(j_max / resolution_x + 0.5)
        # if there is an overlap in the obstacle, we don't consider it
        obstacle_count_grid[i_min_idx:i_max_idx, j_min_idx:j_max_idx] = 1

    return obstacle_count_grid


def _get_obstacle_density(
    measurement_grid_size: list[int],
    obstacles: list,
    x_offset: float,
    y_offset: float,
    grid_resolution: list[float] = DEFAULT_GRID_RESOLUTION,
    R: float = 0.7,
) -> np.ndarray:
    """
    Computes the density of the obstacles at a given point z
    """

    obstacle_density_grid = np.full(measurement_grid_size, 0.0)

    # shift obstacle coordinates
    occupied_areas = _get_occupied_areas(obstacles, x_offset, y_offset)

    # handle the resolution
    resolution_x, resolution_y = grid_resolution[0], grid_resolution[1]

    # apply the gaussian kernel for each obstacle
    for i_min, i_max, j_min, j_max in occupied_areas:
        # iterate over all grid cells and apply the gaussian kernel for current obstacle
        for i in range(obstacle_density_grid.shape[0]):
            for j in range(obstacle_density_grid.shape[1]):
                # for each cell, use the center point of the cell as integration point
                # otherwise, the density is shifted towards the left (grid axes)
                integration_point = [(j + 0.5) * resolution_x, (i + 0.5) * resolution_y]
                integral = integrate.dblquad(
                    lambda x, y: _gauss_density(np.array([x, y]), np.array(integration_point), R),  # noqa: B023
                    i_min,
                    i_max,
                    j_min,
                    j_max,  # integration limits
                )[0]  # only need the result value
                obstacle_density_grid[i, j] += integral

    return obstacle_density_grid


def _get_agent_density(
    grid_data: np.ndarray,
    grid_resolution: list[float] = DEFAULT_GRID_RESOLUTION,
    R: float = DEFAULT_R,
    Sp: float | None = None,
) -> np.ndarray:
    """
    Applies the gaussian kernel from Seitz et.al 2012 to a 2D grid data
    Uses broadcasting to speed up the computation

    NOTE: In contrast to the obstacle density, we do not shift densities to the center of the cell.
    Such a shift is not needed, since the agent density is calculated based on distances only, which are translation invariant.
    """

    # default scaling factors
    if Sp is None:
        Sp = (0.4**2) * np.sqrt(3) / 2  # with 0.4 the average torso diameter

    # get all idxs of the grid cells
    all_idxs = np.array(np.where(np.isfinite(grid_data))).T

    # adjust for the true grid resolution (instead of idxs)
    resolution = list(reversed(grid_resolution))

    # get the agent idxs, i.e. the cells with count >= 1
    agent_idx = np.array(np.where(grid_data >= 1)).T
    agent_idxs_scaled = agent_idx * resolution
    R_sq = 2 * R**2
    coeff = 1 / (np.pi * R_sq)

    all_idxs_scaled = all_idxs * resolution
    diffs = all_idxs_scaled[:, np.newaxis, :] - agent_idxs_scaled  # per cell, the distance to each agent is calculated

    distances = np.sum(diffs**2, axis=2)  # per cell, one entry is calculated with euclidian distance in 2d grid
    # distances = np.sqrt(distances)
    kernel_values = coeff * np.exp(
        -distances / R_sq
    )  # for each cell with count >= 1, we have a full grid of kernel values

    # get the scaling values for each kernel
    kernel_count = grid_data[agent_idx[:, 0], agent_idx[:, 1]]  # number of kernels for agent count

    # write the values in the correct grid cells (back to idx)
    gaussian_grid = np.zeros_like(grid_data, dtype=np.float64)
    gaussian_grid[all_idxs[:, 0], all_idxs[:, 1]] = np.sum(kernel_count * kernel_values, axis=1) * Sp

    return gaussian_grid


def _get_agent_count_grid(df: pd.DataFrame, ts_grid_shape: tuple[int, int, int]) -> np.ndarray:
    """
    Transfer the grid count values into the numpy grid.
    """
    grid = np.full(ts_grid_shape, 0.0)
    data_values = df[['timeStep', 'x_idx', 'y_idx', 'count']].to_numpy()
    for t, x_idx, y_idx, column_value in data_values:
        t, x_idx, y_idx = int(t), int(x_idx), int(y_idx)
        grid[t, y_idx, x_idx] += column_value  # TODO: do we need to + here? Values were already aggregated before ?

    return grid


def _get_agent_density_grid(
    df: pd.DataFrame,
    ts_grid_shape: tuple[int, int, int],
    R: float = DEFAULT_R,
    grid_resolution: list[float] = DEFAULT_GRID_RESOLUTION,
) -> np.ndarray:
    """
    Computes the agent density grid for the given data.
    Is based on the count per cell to scale the density values efficiently
    """
    # transfer the grid count values into the numpy grid
    grid = _get_agent_count_grid(df, ts_grid_shape)

    # need full count grid for density calculation
    agent_density_grid = np.full(ts_grid_shape, 0.0)

    for t in range(ts_grid_shape[0]):  # iterate over all time steps
        agent_density_grid[t] = _get_agent_density(grid[t], grid_resolution=grid_resolution, R=R)

    return agent_density_grid


# Gaussian kernel
def _gaussian_kernel(u):
    """
    Gaussian kernel function.
    Returns the exponential decay based on the squared input distance.
    """
    return np.exp(-(u**2))


# general kernel function
def _kernel(data_point, cell_center, bandwidth, K):
    """
    Computes the kernel value between a point `x` and center `c` using a given kernel function `K`.
    Normalizes the distance by the bandwidth.
    """

    u = np.linalg.norm(cell_center - data_point) / bandwidth
    return K(u)


def _compute_convolved_velocity(
    velocity_grid: np.ndarray,
    count_grid: np.ndarray,
    grid_resolution: list[float] = DEFAULT_GRID_RESOLUTION,
    neighborhood_depth: int = 1,
    kernel=_gaussian_kernel,
    kernel_bandwidth: float = DEFAULT_R,
):
    """Get the mean velocity for each cell and then concolve based on a kernel grid of a given neighborhood depth."""

    # create convolution weight kernel given the neighborhood
    x_axs = np.arange(-(neighborhood_depth), neighborhood_depth + 1, 1) * grid_resolution[0]
    y_axs = np.arange(-(neighborhood_depth), neighborhood_depth + 1, 1) * grid_resolution[1]
    xx, yy = np.meshgrid(x_axs, y_axs, indexing='ij')
    xxyy = np.concatenate([np.expand_dims(xx, axis=2), np.expand_dims(yy, axis=2)], axis=2)

    # create kernel based filter grid
    kernel_grid = np.apply_along_axis(
        lambda center: _kernel(data_point=np.array([0, 0]), cell_center=center, bandwidth=kernel_bandwidth, K=kernel),
        2,
        xxyy,
    )
    # kernel_grid = kernel_grid / np.sum(kernel_grid)

    # get the number of neighbors per cell per step
    neighborhood_mask = np.ones_like(kernel_grid)
    for t, (v_grid, c_grid) in enumerate(zip(velocity_grid, count_grid)):
        # input is extended by filling all values beyond the edge with 0
        timestep_velocity_grid = convolve(input=v_grid, weights=kernel_grid, mode='constant', cval=0)
        neighboor_count_grid = convolve(input=c_grid, weights=neighborhood_mask, mode='constant', cval=0)
        # normalize when there are > 0 neighbors
        velocity_grid[t, :, :] = np.divide(
            timestep_velocity_grid,
            neighboor_count_grid,
            out=np.zeros_like(timestep_velocity_grid),
            where=neighboor_count_grid != 0,
        )

    return velocity_grid


def _get_agent_velocity_grid(df: pd.DataFrame, ts_grid_shape: tuple[int, int, int]) -> np.ndarray:
    """
    Transfer the grid count values into the numpy grid.
    """
    velocity_grid = np.full(ts_grid_shape, 0.0)

    # transfer the grid count values into the numpy grid
    # # extracing values first makes things 20x faster
    data_values = df[['timeStep', 'x_idx', 'y_idx', 'velocity']].to_numpy()
    for t, x_idx, y_idx, velocity in data_values:
        t, x_idx, y_idx = int(t), int(x_idx), int(y_idx)
        velocity_grid[t, y_idx, x_idx] += velocity

    return velocity_grid


def _reduce_grid_df(
    data_dict: dict[int, pd.DataFrame],
    xlim: list[float | None],
    ylim: list[float | None],
    tlim: list[int | None],
    grid_resolution: list[float] = DEFAULT_GRID_RESOLUTION,
) -> tuple[dict[int, pd.DataFrame], list[int]]:
    """
    Filter the data based on the given limits and reduce it to the measurement area.
    The data is shifted by the given offsets for time and space.
    NOTE: at timestep zero there may be no pedestrians present. In this case
    we will obtain empty frames when writing the data to the grid later on and still start at 0.
    """
    # handle the resolution
    resolution_x, resolution_y = grid_resolution[0], grid_resolution[1]
    reduced_data_dict: dict[int, pd.DataFrame] = {}

    # clean up the individual df entries and shift the x and y appropriately
    for key, run_data in data_dict.items():
        # use ceil to ensure that we shift to full cells
        # filter and shift on x coordinates
        if xlim[1] is not None:
            # need to have < xlim[1] to not overflow the last cell
            run_data = run_data[(run_data['x'] < xlim[1])]

        if xlim[0] is not None:
            run_data = run_data[(run_data['x'] >= xlim[0])]
            run_data['x'] = run_data['x'] - xlim[0]  # shift to start at 0.0

        # filter and shift on y coordinates
        if ylim[1] is not None:
            # need to have < xlim[1] to not overflow the last cell
            run_data = run_data[(run_data['y'] < ylim[1])]

        if ylim[0] is not None:
            run_data = run_data[(run_data['y'] >= ylim[0])]
            run_data['y'] = run_data['y'] - ylim[0]

        # discretize the grid for the (starting) position for the corresponding time step
        run_data['x_idx'] = (run_data['x'] / resolution_x).astype(int)
        run_data['y_idx'] = (run_data['y'] / resolution_y).astype(int)

        # write it back to the result dict
        reduced_data_dict[key] = run_data

    # get the min and max cell indices where pedestrians are present AFTER adjusting the observation area
    min_x_index = min([run_data['x_idx'].min() for run_data in reduced_data_dict.values()])
    min_y_index = min([run_data['y_idx'].min() for run_data in reduced_data_dict.values()])
    max_x_index = max([run_data['x_idx'].max() for run_data in reduced_data_dict.values()])
    max_y_index = max([run_data['y_idx'].max() for run_data in reduced_data_dict.values()])

    # overwrite the x and y max cell indices in case the data was filtered using xlim and ylim
    # not always only using the area with pedestrians present allows to keep some empty boundary areas
    # NOTE: this "keeping" may lead to boundary effects in case the area of trajectory start and end is not cut of properly
    if xlim[1] is not None:
        x_upper_bound = xlim[1] - xlim[0] if xlim[0] is not None else xlim[1]
        if int((x_upper_bound - 0.001) / resolution_x) > max_x_index:
            log.warning(
                f'The largest x-index with pedestrians present is at {max_x_index=}, '
                f'short of the grid edge at {int(x_upper_bound / resolution_x)=}. '
                f'This may lead to boundary effects, so please check {xlim[1]=}'
            )
        max_x_index = int((x_upper_bound - 0.001) / resolution_x)  # move the limit within the cell in edge cases
        log.debug(f'Set {max_x_index=} based on {xlim[1]} after cutting data accordingly.')

    if ylim[1] is not None:
        y_upper_bound = ylim[1] - ylim[0] if ylim[0] is not None else ylim[1]
        if int((y_upper_bound - 0.001) / resolution_y) > max_y_index:
            log.warning(
                f'The largest y-index with pedestrians present is at {max_y_index=}, '
                f'short of the grid edge at {int(y_upper_bound / resolution_y)=}. '
                f'This may lead to boundary effects, so please check {ylim[1]=}'
            )
        max_y_index = int((y_upper_bound - 0.001) / resolution_y)  # move the limit within the cell in edge cases
        log.debug(f'Set {max_y_index=} based on {ylim[1]} after cutting data accordingly.')

    if min_x_index > 0:
        log.warning(
            f'The lowest x-index with pedestrians present is at {min_x_index=}, not at the grid '
            f'edge. This may lead to boundary effects, so please check {xlim[0]=}'
        )

    if min_y_index > 0:
        log.warning(
            f'The lowest y-index with pedestrians present is at {min_y_index=}, not at the grid '
            f'edge. This may lead to boundary effects, so please check {ylim[0]=}'
        )

    # apply tlims for all series
    # if tlim is not set, time series may have a different length
    for key, run_data in reduced_data_dict.items():
        if tlim[0] is not None:
            run_data = run_data[(run_data['timeStep'] >= tlim[0])]
            run_data.loc[:, 'timeStep'] = run_data['timeStep'] - tlim[0]

        if tlim[1] is not None:
            # filter on max timestep
            run_data = run_data[(run_data['timeStep'] <= tlim[1])]

        # write it back to the result dict
        reduced_data_dict[key] = run_data

    log.info(f'Applied {tlim=} to each dataset separately.')
    log.info(
        f'Maximum timestep per series id: {[(key, run_data["timeStep"].max()) for key, run_data in reduced_data_dict.items()]}'
    )

    # set the measurement grid size after everything was adjusted and checked :)
    measurement_grid_size = [int(max_y_index + 1), int(max_x_index + 1)]

    return reduced_data_dict, measurement_grid_size
