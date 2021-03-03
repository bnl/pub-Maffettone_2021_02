"""
Recreate Figures 7 and 8 from the paper
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from bad_seeds.plot.gen_figs import plot_all_ideal, plot_all_timelimit


def main():
    figsize = (8.5 / 2.54, (8.5 / 2.54) / 1.6)
    batch_sizes = (1, 8, 16, 32, 64, 128, 256, 512)
    timelimits = (10, 20, 30, 40, 50, 70, 100)
    data_path = Path().absolute() / "published_results"
    out_path = Path().absolute()

    with mpl.rc_context({"font.size": 7}):
        fig, axes_ideal = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_all_ideal(
            ax=axes_ideal,
            data_path=data_path,
            batch_sizes=batch_sizes,
        )
        fig.savefig(out_path / "figure_7.png", dpi=300)
        print(f"wrote {out_path / 'figure_7.png'}")
        fig, axes_timelimited = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_all_timelimit(
            ax=axes_timelimited,
            data_path=data_path,
            timelimits=timelimits,
        )
        fig.savefig(out_path / "figure_8.png", dpi=300)
        print(f"wrote {out_path / 'figure_8.png'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recreate figures from paper.")
    parser.add_argument(
        "--show", action="store_true", help="Show the figures for interactive viewing"
    )
    args = parser.parse_args()
    if not args.show:
        mpl.use("agg")
    main()
    if args.show:
        plt.show()
