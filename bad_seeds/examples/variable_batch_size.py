"""
A script for a short run of the experiments needed to generate the comparative batch results.
For the full results, more episodes should be used. These results are presented in the published_results folder
for convenience.
"""
from collections import namedtuple
from pathlib import Path
from bad_seeds.training import train_a2c_cartseed
from bad_seeds.utils.tf_utils import csv_from_accumulator


def main(simple=True):
    """
    Main to run experiments comparing batchsize. Defaults to a simple experiment that is unlikely to converge;
    however, will demonstrate functionality.
    `simple = False` will give a  complete, albeit expensive, experiment.

    This will output a .csv file of results for each batch size in an `./example_results` directory.

    Parameters
    ----------
    simple: bool
        Whether or not a simple, truncated, low cost experiment is run as a demonstration

    Returns
    -------

    """
    SetupArgs = namedtuple("SetupArgs", "batch_size env_version out_path num_episodes")
    if simple:
        batch_sizes = (64, 512)
        num_episodes = 250
    else:
        batch_sizes = (1, 8, 16, 32, 64, 128, 256, 512)
        num_episodes = int(3 * 10 ** 3)

    out_path = Path().absolute() / "example_results"
    run_args = {
        f"batch_{i}": SetupArgs(i, 2, out_path, num_episodes) for i in batch_sizes
    }
    for key, setup_args in run_args.items():
        print(f"{key}: {setup_args}")
        train_a2c_cartseed.main(**setup_args._asdict())
        summary_path = sorted(
            (
                out_path
                / "training_data"
                / "a2c_cartseed"
                / f"{setup_args.env_version}_None_default_{setup_args.batch_size}"
            ).glob("summary-*")
        )[0]
        csv_from_accumulator(summary_path, csv_path=out_path / f"{key}.csv")


if __name__ == "__main__":
    main()
