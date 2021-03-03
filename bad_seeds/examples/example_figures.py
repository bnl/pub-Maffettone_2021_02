"""
Recreate equivalents of Figures 7 and 8 from the paper, but with
example results.

"""
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from bad_seeds.plot.gen_figs import plot_all_ideal, plot_all_timelimit


def main():
    figsize = (8.5 / 2.54, (8.5 / 2.54) / 1.6)
    data_path = Path().absolute() / "example_results"
    out_path = Path().absolute()

    batch_sizes = [int(f.stem.partition("_")[2]) for f in data_path.glob("batch*.csv")]
    timelimits = defaultdict(list)
    for f in data_path.glob("*default*.csv"):
        time, _, batch = f.stem.split("_")
        timelimits[int(batch)].append(int(time))

    with mpl.rc_context({"font.size": 7}):
        fig, axes_ideal = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_all_ideal(
            ax=axes_ideal,
            data_path=data_path,
            batch_sizes=batch_sizes,
        )
        fig.savefig(out_path / "ideal_training.png", dpi=300)
        print(f"wrote {out_path / 'ideal_training.png'}")
        for batch_size, _timelimits in timelimits.items():
            fig, axes_timelimited = plt.subplots(
                figsize=figsize, constrained_layout=True
            )
            plot_all_timelimit(
                ax=axes_timelimited,
                data_path=data_path,
                timelimits=_timelimits,
                batch_size=batch_size,
            )
            fig.text(0, 0, f"Batch size = {batch_size}")
            save_name = out_path / f"time_constrained_{batch_size}.png"
            fig.savefig(save_name, dpi=300)
            print(f"wrote {save_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="""Recreate figures from paper on locally generated data.

Attempts to infer the generated parameters by parsing the file names
in example_results.  This is brittle and may not work in all cases.

"""
    )
    parser.add_argument(
        "--show", action="store_true", help="Show the figures for interactive viewing"
    )
    args = parser.parse_args()
    if not args.show:
        mpl.use("agg")
    main()
    if args.show:
        plt.show()
