"""
Writing figures and animations to the run folder.

Every figure this package produces lands in one of four subdirectories of the
run folder -- ``data/``, ``evaluation/``, ``interpretation/``, ``gifs/`` -- and
takes its file type and resolution from the run's :class:`~.style.PlotStyle`.
Both functions here take that style instead of a path/filetype/dpi triple, so
no plotting function has to know how the run folder is laid out.
"""

import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

from .config import DEFAULT_GIF_FPS, get_animation_writer
from .style import PlotStyle

log = logging.getLogger(__name__)


def save_figure(
    fig: Figure,
    style: PlotStyle,
    category: str,
    name: str,
    close_after: bool = True,
    bbox_inches: Optional[str] = 'tight',
    pad_inches: Optional[float] = None,
) -> None:
    """
    Save a figure as ``<style.path>/<category>/<name>.<style.fig_filetype>``.

    With ``style.path`` set to None the figure is only closed, not written --
    that is what makes a plotting function callable from a notebook without
    leaving files behind.

    Parameters
    ----------
    fig : Figure
        The figure to save.
    style : PlotStyle
        Supplies the run folder, file type and dpi.
    category : str
        Subdirectory: 'data', 'evaluation', 'interpretation' or 'gifs'.
    name : str
        File name *without* extension.
    close_after : bool, default=True
        Close the figure once it is written.
    bbox_inches, pad_inches
        Passed straight to ``savefig``.

    Examples
    --------
    >>> save_figure(fig, style, 'evaluation', 'density_mae_train')
    """
    if style.path is not None:
        full_path = os.path.join(style.path, category, f'{name}.{style.fig_filetype}')
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        savefig_kwargs = {'dpi': style.dpi, 'bbox_inches': bbox_inches}
        if pad_inches is not None:
            savefig_kwargs['pad_inches'] = pad_inches

        fig.savefig(full_path, **savefig_kwargs)
        log.info(f'Saved figure to {full_path}')
    else:
        log.debug('Style has no path, skipping save')

    if close_after:
        plt.close(fig)


def save_animation(
    ani: FuncAnimation,
    fig: Figure,
    style: PlotStyle,
    name: str,
    fps: int = DEFAULT_GIF_FPS,
    close_after: bool = True,
) -> Optional[str]:
    """
    Save an animation as ``<style.path>/gifs/<name>.<style.gif_filetype>``.

    The writer is chosen by :func:`~.config.get_animation_writer`, which falls
    back to a GIF when the requested format needs a tool that is not installed
    -- so the extension actually written is the one returned here, not
    necessarily ``style.gif_filetype``.

    Returns
    -------
    str or None
        The path written, or None when the style has no path.
    """
    if style.path is None:
        log.debug('Style has no path, skipping animation save')
        if close_after:
            plt.close(fig)
        return None

    writer, actual_filetype = get_animation_writer(style.gif_filetype, fps)

    gif_directory = os.path.join(style.path, 'gifs')
    os.makedirs(gif_directory, exist_ok=True)
    output_path = os.path.join(gif_directory, f'{name}.{actual_filetype}')

    log.info(f'Saving animation to {output_path} using {writer.__class__.__name__}')
    ani.save(output_path, dpi=style.dpi, writer=writer)
    log.info(f'Saved animation to {output_path}')

    if close_after:
        plt.close(fig)

    return output_path
