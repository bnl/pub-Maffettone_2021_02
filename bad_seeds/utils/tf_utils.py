import tensorflow as tf
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
import pandas as pd
import logging


def tensorflow_settings(gpu_idx=0):
    """
    Convenience function for setting up tensorflow
    Parameters
    ----------
    gpu_idx: int
        Index of GPU to be used from list of physical GPU devices

    Returns
    -------
    None
    """
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        # Restrict TensorFlow to a specific GPU, and dynamically grow memory use
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_idx], "GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


def load_accumulator(path):
    """Extract data from tensorboard summaries"""
    event_acc = EventAccumulator(str(path), size_guidance={"tensors": 0})
    event_acc.Reload()
    w_times, step_nums, vals = zip(*event_acc.Tensors("agent.observe/episode-reward"))
    return w_times, step_nums, [float(tf.make_ndarray(a)) for a in vals]


def csv_from_accumulator(path, csv_path=None):
    """
    Construct a csv from a tensorboard summary,
    placing the csv in the parent's parent directory, with the same name as the parent
    Parameters
    ----------
    path : str, Path
        Tensorboard summary directory
    csv_path : Path
        Output csv directory

    Returns
    -------

    """
    if csv_path is None:
        csv_path = path.parent.parent / Path(path.parent.name + ".csv")
    else:
        csv_path = Path(csv_path).expanduser()
    w_times, step_nums, vals = load_accumulator(path)
    df = pd.DataFrame({"wall": w_times, "step": step_nums, "val": vals})
    df.to_csv(str(csv_path), index=False)
