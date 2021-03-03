"""
Plotting functions for Figures in bad seed paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pathlib import Path


def smooth(scalars, weight):  # Weight between 0 and 1
    """Apply an exponential smoothing window

    Parameters
    ----------
    scalars : (N, ) array
        The time series to smooth.

    weight : float [0, 1]
        The smoothing weight.

    Returns
    -------
    smoothed : (N, ) array
        The smoothed data, same size as input data.
    """
    # First value in the plot (first timestep)
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        # Calculate smoothed value
        smoothed_val = last * weight + (1 - weight) * point
        # Save it
        smoothed.append(smoothed_val)
        # Anchor the last smoothed value
        last = smoothed_val

    return np.array(smoothed)


def general_axis_adjustments(ax, x_max):
    """Set standard axis labels and limits

    Parameters
    ----------
    ax : matplotlib.axes.Axes
       The Axes to adjust the axis of

    x_max : number
       The x-max for this Axes

    Returns
    -------
    ax : matplotlib.axes.Axes
       Same object passed in
    """
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 110, 10))
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Normalized score")
    return ax


def plot_timelimit_learning(df, timelimit, *, ax, label_loc="baseline", **kwargs):
    """Draw a single line for the time-limited figure.

    Parameters
    ----------
    df : pd.DataFrame

    timelimit : int
        The timelimit of this experiment in cycles

    ax : matplotlib.ax.Axes
        The axes to plot to.

    label_loc : {'baseline', 'data'}, optional
        Place the label at the baseline of the data or where it hits the
        right axes

    Other Parameters
    ----------------
    **kwargs
       Additional keyword arguments are passed through to ax.plot

    Returns
    -------
    ln, axline : matplotlib.lines.Line2D
    ann : matplotlib.text.Annotation
    ax : matplotlib.axes.Axes
    """
    values = smooth(df.val, 0.997)
    (ln,) = ax.plot(df.step, values, label=f"Fixed time = {timelimit}", **kwargs)

    if label_loc == "data":
        end_y = np.mean(values[-15:])
    elif label_loc == "baseline":
        end_y = 90 * timelimit / 100
    else:
        raise ValueError("label_loc must be one of {'data', 'baseline'}")

    axline = ax.axhline(end_y, ls="--", label=f"Sequential, t = {timelimit}", **kwargs)
    ann = ax.annotate(
        f"{timelimit} turns",
        (1, end_y),
        xycoords=ax.get_yaxis_transform(),
        xytext=(3, 0),
        ha="left",
        va="center",
        textcoords="offset points",
        weight="bold",
        **kwargs,
    )
    return ln, axline, ann


def plot_all_timelimit(
    data_path,
    timelimits,
    ax,
    *,
    l_alpha=0.9,
    score="default",
    batch_size=512,
):
    """Make the time-limited panel

    This expect that there will be CSV files in data_path with names ::

       batch_{batch_size}.csv

    Parameters
    ----------
    data_path : Path, optional
        The location of the data files
    timelimits : List[int]
        The timelimits of the data to be plotted.
    ax : matplotlib.axes.Axes
        The axes to plot to
    l_alpha : float [0, 1], optional
        The alpha ot use drawing the lines
    score : str, optional
        The scoring mode used.  Second value in name template.
    batch_size : int, optional
        The training batch size.  Third value in name template.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """

    cmap = plt.get_cmap("Dark2")
    max_step = -1
    plotted_data = {}
    for j, timelimit in enumerate(sorted(timelimits, reverse=True)):
        path = data_path / Path(f"{timelimit}_{score}_{batch_size}.csv")
        df = pd.read_csv(str(path))
        key = (timelimit, score, batch_size)
        plotted_data[key] = plot_timelimit_learning(
            df, timelimit, ax=ax, alpha=l_alpha, color=cmap(j)
        )
        max_step = max(max_step, np.max(df.step))

    ax.legend(
        handles=(
            plt.Line2D([], [], color="k", ls="--", label="Sequential", alpha=l_alpha),
            plt.Line2D([], [], color="k", ls="-", label="Agent", alpha=l_alpha),
        ),
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=3,
        mode="expand",
        borderaxespad=0.0,
        frameon=False,
    )
    ax = general_axis_adjustments(ax, max_step)
    return plotted_data


def plot_all_ideal(batch_sizes, ax, l_alpha=0.9, *, vmin=1, vmax=512, data_path):
    """Make the time-limited panel

    This expect that there will be CSV files in data_path with names ::

       {timelimit}_{score}_{batch_size}.csv

    Parameters
    ----------
    timelimits : List[int]
        The timelimits of the data to be plotted.

    ax : matplotlib.axes.Axes
        The axes to plot to

    l_alpha : float [0, 1], optional
        The alpha ot use drawing the lines
    data_path : Path, optional
        The location of the data files
    score : str, optional
        The scoring mode used.  Second value in name template.
    batch_size : int, optional
        The training batch size.  Third value in name template.

    Returns
    -------
    plotted_data : Dict[int, Line2D]
    ideal_p : Line2D
    seq_p : Line2D
    cbar : ColorBar
    """

    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "Dan hates Yellow", plt.get_cmap("viridis_r")(np.linspace(0.2, 1, 256))
    )
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    plotted_data = {}
    max_step = -1
    for batch_size in batch_sizes:
        path = data_path / Path(f"batch_{batch_size}.csv")
        df = pd.read_csv(str(path))

        (plotted_data[batch_size],) = ax.plot(
            df.step,
            smooth(df.val, 0.997),
            alpha=l_alpha,
            color=cmap(norm(batch_size)),
            label=f"Batch size = {batch_size}",
        )
        max_step = max(max_step, np.max(df.step))

    ideal_p = ax.axhline(90, color="k", ls=":", label="Ideal")
    seq_score = 90 * np.mean([i / 10 for i in range(1, 11)])
    seq_p = ax.axhline(seq_score, color="k", ls="--", label="Sequential")

    cbar = ax.figure.colorbar(sm, ax=ax, label="Batch Size")
    ax.legend(
        handles=(
            ideal_p,
            seq_p,
            plt.Line2D([], [], color="k", ls="-", label="Agent", alpha=l_alpha),
        ),
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=3,
        mode="expand",
        borderaxespad=0.0,
        frameon=False,
    )
    ax = general_axis_adjustments(ax, np.max(df.step))
    return plotted_data, ideal_p, seq_p, cbar


def make_figure(
    data_path=Path("published_results"),
    *,
    figsize=(8.5 / 2.54, 5),
    out_file=None,
    batch_sizes=(32, 64, 128, 512),
    ideal_kwargs=None,
    timelimits=(10, 30, 70, 100),
    timelimited_kwargs=None,
):
    """Generates Figure N from the paper.

    Parameters
    ----------
    data_path : Path
        The folder to find the data files in.

        This expect that there will be CSV files in data_path with names ::

           {timelimit}_{score}_{batch_size}.csv

        for the timelimited data and ::

           batch_{batch_size}.csv

        for the ideal data.

    figsize : (float, float), optional
    out_file : Path, optional
        If not None, save the figure.
    batch_sizes : Tuple[int], optional
        The batches sizes to plot in the top panel
    ideal_kwargs : dict, optional
        Any other kwargs to pass through to `plot_all_ideal`
    timelimits : Tuple[int], optional
        The timelimits to use for the bottom panel.
    timelimited_kwargs : dict, optional
        Any other kwargs to pass through to `plot_all_timelimit`

    """
    with mpl.rc_context({"font.size": 7}):
        fig, (axes_ideal, axes_timelimited) = plt.subplots(
            2, 1, figsize=figsize, constrained_layout=True
        )
        arts_ideal = plot_all_ideal(
            ax=axes_ideal,
            data_path=data_path,
            batch_sizes=batch_sizes,
            **(ideal_kwargs or {}),
        )
        arts_timelimited = plot_all_timelimit(
            ax=axes_timelimited,
            data_path=data_path,
            timelimits=timelimits,
            **(timelimited_kwargs or {}),
        )
    if out_file is not None:
        fig.savefig(out_file, dpi=300)

    return (
        fig,
        {"ideal": axes_ideal, "timelimited": axes_timelimited},
        {"ideal": arts_ideal, "timelimited": arts_timelimited},
    )
