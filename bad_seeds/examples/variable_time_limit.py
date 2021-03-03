"""
A script for a short run of the experiments needed to generate the comparative time limit results.
For the full results, more episodes should be used. These results are presented in the published_results folder
for convenience.
"""
from collections import namedtuple
from pathlib import Path
from bad_seeds.training import train_a2c_cartseed
from bad_seeds.utils.tf_utils import csv_from_accumulator


def main(simple=True):
    """
    Main to run experiments comparing time limitations. Defaults to a simple experiment that is unlikely to converge;
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
    SetupArgs = namedtuple(
        "SetupArgs", "time_limit batch_size env_version out_path num_episodes"
    )
    if simple:
        time_limits = [20, 40, 70]
        num_episodes = 250
    else:
        time_limits = list(range(10, 110, 10))
        num_episodes = int(3 * 10 ** 3)

    batch_size = 16
    out_path = Path().absolute() / "example_results"
    run_args = {
        f"{i}_default_{batch_size}": SetupArgs(i, batch_size, 2, out_path, num_episodes)
        for i in time_limits
    }
    for key, setup_args in run_args.items():
        print(f"{key}: {setup_args}")
        train_a2c_cartseed.main(**setup_args._asdict())
        summary_path = sorted(
            (
                out_path
                / "training_data"
                / "a2c_cartseed"
                / f"{setup_args.env_version}_{setup_args.time_limit}_default_{setup_args.batch_size}"
            ).glob("summary-*")
        )[0]
        csv_from_accumulator(summary_path, csv_path=out_path / f"{key}.csv")


if __name__ == "__main__":
    main()
