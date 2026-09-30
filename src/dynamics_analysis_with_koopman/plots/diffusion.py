"""
Diffusion map visualization module.

This module provides functions for visualizing diffusion maps and embeddings,
including:
- Diffusion map coordinate time series plots
- Eigenvalue visualization
- 3D scatter plots with animated GIF generation
- Per-series embedding analysis
"""

import io
import logging
import math
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image

from ..config.configs import FullConfig
from .config import SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE
from .layout import apply_standard_layout, calculate_legend_position
from .io_utils import save_figure
from .style import PlotStyle

log = logging.getLogger(__name__)


def visualize_diffusion_maps(
    run_config: FullConfig,
    run_metrics,
    diff_maps_df: pd.DataFrame,
    diff_embedding_df: pd.DataFrame,
    store_visualizations: bool = True,
) -> None:
    """
    Visualize the diffusion maps and the corresponding eigenvalues.

    Entry point for the pipeline: builds the run's :class:`PlotStyle` once and
    calls the individual plotting functions with it.

    Parameters
    ----------
    run_config : FullConfig
        Configuration object containing run parameters and paths
    run_metrics : RunMetrics
        Run metrics, for the timesteps and grid geometry
    diff_maps_df : pd.DataFrame
        DataFrame containing diffusion map eigenvectors and eigenvalues
    diff_embedding_df : pd.DataFrame
        DataFrame containing embedded data points
    store_visualizations : bool, default=True
        Whether to save the individual 3d frames and their GIF to disk
    """
    if diff_embedding_df is None or diff_maps_df is None:
        return

    log.info('Create plots for diffusion maps.')
    style = PlotStyle.from_run(run_config, run_metrics)

    plot_diffusion_map_timeseries(df=diff_embedding_df, style=style)
    plot_diffusion_map_evs(df_evecs=diff_maps_df, style=style)

    # create gif from the frames and store it in checkpoint directory
    try:
        fig = scatter3d_with_gif(
            diff_embedding_df,
            style=style,
            store_visualizations=store_visualizations,
        )
    except Exception as e:
        log.error('Could not create gif from diffusion map frames. Continue...')
        log.error(f'Error: {e}')
        return

    if fig is not None:
        fig.show()


def _draw_coordinate_panels(df: pd.DataFrame, axs, style: PlotStyle) -> None:
    """One panel per diffusion map coordinate, with one line per series."""
    for series_id in df['ID'].unique():
        subset_df = df[df['ID'] == series_id]
        time_in_s = subset_df['time'] * style.temporal_resolution

        col_names = [col for col in subset_df.columns if 'diff_coord_' in col]
        for col_name, ax in zip(col_names, axs.flatten()):
            ax.plot(time_in_s, subset_df[col_name], label=f'time series {series_id}')
            ax.set_title(f'Diffusion map coordinate {col_name.split("_")[-1]} over time', fontsize=MEDIUM_SIZE)
            ax.set_xlabel('Time in seconds', fontsize=SMALL_SIZE)
            ax.set_ylabel('Coordinate value', fontsize=SMALL_SIZE)
            ax.label_outer()  # Only show outer labels to avoid clutter

            # mark the timesteps the grid figures show
            for timestep in style.visualization_timesteps:
                ax.axvline(
                    timestep * style.temporal_resolution, color='black', linestyle='--', alpha=0.5, linewidth=0.7
                )


def _add_bottom_legend(fig, axs, num_items: int) -> None:
    """Put one shared legend below the grid, making room for it first."""
    legend_config = calculate_legend_position(num_items=num_items, position='bottom')

    if legend_config['bottom_adjust'] is not None:
        fig.subplots_adjust(bottom=legend_config['bottom_adjust'])

    axs.flat[-1].legend(
        loc=legend_config['loc'],
        bbox_to_anchor=legend_config['bbox_to_anchor'],
        ncol=legend_config['ncol'],
        frameon=legend_config['frameon'],
    )


def plot_diffusion_map_timeseries(
    df: pd.DataFrame, style: PlotStyle, is_prediction: bool = False, file_name: Optional[str] = None
) -> None:
    """
    Plot diffusion map coordinate time series for all series.

    Creates a large grid showing how each diffusion map coordinate evolves over
    time across all time series. Useful for understanding temporal dynamics in
    the embedding space.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'ID', 'time', and 'diff_coord_N' columns
    style : PlotStyle
        Supplies the time axis, the marked timesteps and output settings
    is_prediction : bool, default=False
        Whether this is prediction data (affects title and filename)
    file_name : str, optional
        Custom filename (without extension). If None, uses default naming.
    """
    if is_prediction:
        postfix = 'pred'
    else:
        postfix = 'truth'

    log.info('Plot diffusion map coordinate time series for all series.')

    # plot one subfigure for each diffusion map coordinate time series
    nrows = math.ceil((len(df.columns) - 2) / 10)
    fig, axs = plt.subplots(ncols=10, nrows=nrows, figsize=(65, nrows * 5))  # Increased from 3 to 5

    # Add suptitle FIRST (key to avoiding overlap)
    fig.suptitle(
        f'Diffusion map coordinate per time series: {"Prediction" if is_prediction else "Ground truth"} embedding',
        fontsize=BIGGER_SIZE,
    )

    _draw_coordinate_panels(df, axs, style)

    # Reduced padding between rows compared to the default
    apply_standard_layout(
        fig,
        has_suptitle=True,
        has_colorbar=False,
        num_rows=nrows,
        h_pad=1.5,  # Reduced padding between rows (default would be max(3.0, 1.0 + nrows * 1.0))
    )

    _add_bottom_legend(fig, axs, num_items=len(df['ID'].unique()))

    name = file_name or f'diffusion_map_coordinate_series_{postfix}'
    save_figure(fig, style, 'interpretation', name)

    log.info(f'Plot saved to interpretation/{name}')


def plot_diffusion_map_evs(
    df_evecs: pd.DataFrame,
    style: PlotStyle,
) -> None:
    """
    Plot the eigenvalues of the diffusion map.

    Creates a simple line plot showing the eigenvalue spectrum, which indicates
    the importance of each diffusion map coordinate.

    Parameters
    ----------
    df_evecs : pd.DataFrame
        DataFrame with columns 'index' and 'eigenvalue'
    style : PlotStyle
        Supplies the output settings
    """
    # plot the eigenvalues as single line
    fig, ax = plt.subplots()
    ax.plot(df_evecs['index'], df_evecs['eigenvalue'], label='Eigenvalues')

    ax.set_title('Eigenvalues of diffusion maps')
    ax.set_xlabel('Diffusion map index')
    ax.set_ylabel('Eigenvalue')
    ax.set_ylim(0, 1.1)
    ax.set_xticks(range(len(df_evecs['eigenvalue'])))
    ax.set_xticklabels([i + 1 for i in df_evecs['index']], rotation=45)

    save_figure(fig, style, 'interpretation', 'diffusion_map_eigenvalues')

    log.info('Saved eigenvalue plot at interpretation/diffusion_map_eigenvalues')


def _set_clusters_and_shapes(
    df: pd.DataFrame,
    diff_map_selection: List[int],
) -> tuple:
    """
    Prepare data for the 3D scatter plot: select columns and add uniform styling.

    Every point gets the same color, marker and label — there is currently no
    per-series or per-time-window distinction (see S19 in the cleanup plan for
    why: the previous version filtered points to a single hardcoded series ID
    and colored by a hardcoded, permanently-empty list of time windows).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with 'ID', 'time', and 'diff_coord_N' columns
    diff_map_selection : list of int
        Three diffusion map coordinates to select (must be length 3)

    Returns
    -------
    tuple of (pd.DataFrame, list of str)
        Enriched DataFrame, and list of selected column names

    Raises
    ------
    ValueError
        If columns don't exist or selection is not exactly 3 coordinates
    """
    diff_map_cols = [f'diff_coord_{i}' for i in diff_map_selection]
    missing_cols = [col for col in diff_map_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f'The requested visualization columns {missing_cols} do not exist in the diffusion map '
            f'coordinate dataframe. Available columns: {list(df.columns)}.'
        )
    if len(diff_map_cols) != 3:
        raise ValueError(
            f'Exactly 3 diffusion map coordinates need to be selected for a 3d plot, got {diff_map_selection}.'
        )

    subset_cols = ['ID', 'time'] + diff_map_cols
    df = df[subset_cols]

    # Uniform styling: no per-series or per-time-window distinction.
    df['color'] = 'gray'
    df['label'] = 'other times'
    df['marker'] = 'circle'

    return df, diff_map_cols


def scatter3d_with_gif(
    df: pd.DataFrame,
    style: PlotStyle,
    diff_map_selection: Optional[List[int]] = None,
    store_visualizations: bool = False,
) -> Optional[go.Figure]:
    """
    Create an animated 3D scatter plot of diffusion map coordinates.

    This function generates an interactive 3D visualization using plotly, where
    points are added incrementally to create an animation. The animation can be
    saved as a GIF.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'ID', 'time', and 'diff_coord_N' columns
    style : PlotStyle
        Supplies the run folder and the run name used in the title
    diff_map_selection : list of int, optional
        Three coordinates to visualize. Default: [1, 2, 4]
    store_visualizations : bool, default=False
        Whether to save individual frames and GIF to disk

    Returns
    -------
    plotly.graph_objects.Figure or None
        The final plotly figure, or None if no data available

    Notes
    -----
    This function uses plotly instead of matplotlib because plotly excels at
    3D interactive visualizations. The animation is created by iteratively
    adding traces and capturing frames.
    """
    if diff_map_selection is None:
        diff_map_selection = [1, 2, 4]

    df, diff_map_cols = _set_clusters_and_shapes(df, diff_map_selection)

    if len(df) == 0:
        log.warning(
            'No data points for the selected series ids. Unable to produce 3d scatter plot for diffusion maps...'
        )
        return None

    selection_id = '_'.join([str(i) for i in diff_map_selection])

    fig = _diffusion_scatter_figure(diff_map_selection, style)
    frames = _add_traces_frame_by_frame(fig, df, diff_map_cols)

    if store_visualizations:
        _store_scatter_frames(frames, selection_id, style)

    return fig


def _diffusion_scatter_figure(diff_map_selection: List[int], style: PlotStyle) -> go.Figure:
    """An empty 3d plotly figure, titled and axis-labelled for the selected coordinates."""
    fig = go.Figure()
    fig.update_layout(
        title=f'3d plot for diffusion map coordinates for run {style.run_name}',
        annotations=[
            dict(
                text=f'Selected coordinates: {diff_map_selection}',
                xref='paper',
                yref='paper',
                x=0.5,
                y=1.05,  # Position above the main title
                xanchor='center',
                yanchor='bottom',
                showarrow=False,
                font=dict(size=16),
            )
        ],
        scene=dict(
            xaxis_title=f'Diffusion emb. coord. {diff_map_selection[0]}',
            yaxis_title=f'Diffusion emb. coord. {diff_map_selection[1]}',
            zaxis_title=f'Diffusion emb. coord. {diff_map_selection[2]}',
        ),
        legend=dict(itemsizing='constant'),
    )
    return fig


def _add_traces_frame_by_frame(
    fig: go.Figure,
    df: pd.DataFrame,
    diff_map_cols: List[str],
) -> List[Image.Image]:
    """
    Add one point cluster at a time, capturing the figure after each.

    The animation is the *build-up* of the scatter plot: every frame has one
    more cluster than the last, so the viewer sees the series appear one by one.
    """
    frames = []

    for i, (symbol, label) in enumerate(
        [(symbol, label) for symbol in df['marker'].unique() for label in df['label'].unique()]
    ):
        subset_df = df[(df['marker'] == symbol) & (df['label'] == label)]

        log.info(f'Frame {i + 1}: Adding points for {symbol} and label {label} ...')

        # the whole cluster shares one color, because they share one label
        series_id = subset_df['ID'].iloc[0]
        log.info(f'... with color {subset_df["color"].iloc[0]}')

        fig.add_trace(
            go.Scatter3d(
                x=subset_df[diff_map_cols[0]],
                y=subset_df[diff_map_cols[1]],
                z=subset_df[diff_map_cols[2]],
                mode='markers',
                text=subset_df['time'],
                marker=dict(
                    symbol=symbol,
                    size=3,
                    color=subset_df['time'],
                    colorscale='viridis_r',  # can't use CMAP, because it is from matplotlib
                    opacity=1.0,
                ),
                showlegend=True,
                name=f'{series_id}: {label}',
            )
        )
        fig.update_layout(showlegend=True)

        # capture the frame straight from memory
        frames.append(Image.open(io.BytesIO(pio.to_image(fig, format='png'))))

    return frames


def _store_scatter_frames(
    frames: List[Image.Image],
    selection_id: str,
    style: PlotStyle,
) -> None:
    """Write the individual frames next to the other figures, and their GIF to gifs/."""
    if not frames:
        log.warning('No frames were captured, so no diffusion map gif is written.')
        return

    image_directory = os.path.join(style.path, 'interpretation')  # exists for sure
    gif_directory = os.path.join(style.path, 'gifs')
    os.makedirs(gif_directory, exist_ok=True)  # may not exist depending on the run settings

    for i, image in enumerate(frames, start=1):
        frame_filename = os.path.join(image_directory, f'diff_map_{selection_id}_frame_{i}.png')
        image.save(frame_filename)
        log.info(f'Frame {i} saved as {frame_filename}')

    gif_path = os.path.join(gif_directory, f'diff_map_{selection_id}.gif')
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],  # Add the rest of the frames
        duration=700,  # Duration per frame in milliseconds
        loop=0,  # Loop indefinitely
    )
    log.info(f'GIF saved as {gif_path}')
