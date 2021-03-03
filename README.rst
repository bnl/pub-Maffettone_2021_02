*********
Bad Seeds
*********
Overview
========
Bad Seeds is a demonstration of using reinforcement learning for optimizing scientific operations when certain samples,
the so-called ‘bad seeds’, can require significantly more measurement time to achieve the desired statistics.
We construct an environment using the `tensorforce <https://github.com/tensorforce/tensorforce>`_ library to emulate
this scenario: where of a given set of samples, unknown to the user *a priori*, there is a randomly distributed
subet of 'bad' samples with a variable 'badness'. We then train an A2C agent to optimally measure these samples,
maximizing the excpected scientific reward.

Details of this work and it's application to beamline science has been published in
`Machine Learning: Science and Technology <https://doi.org/10.1088/2632-2153/abc9fc>`_.
This repository contains the code to reproduce the results contained in that publication. The code demonstrating
how to train a cartpole agent using bluesky and ophyd is presented in the
`bluesky repository <https://github.com/bluesky/bluesky-cartpole>`_.


Abstract
********
Beamline experiments at central facilities are increasingly demanding of remote, high-throughput, and adaptive operation conditions.
To accommodate such needs, new approaches must be developed that enable on-the-fly decision making for data intensive challenges.
Reinforcement learning (RL) is a domain of AI that holds the potential to enable autonomous operations in a feedback loop between beamline experiments and trained agents.
Here, we outline the advanced data acquisition and control software of the Bluesky suite, and demonstrate its functionality with a canonical RL problem: cartpole.
We then extend these methods to efficient use of beamline resources by using RL to develop an optimal measurement strategy for samples with different scattering characteristics.
The RL agents converge on the empirically optimal policy when under-constrained with time.
When resource limited, the agents outperform a naive or sequential measurement strategy, often by a factor of 100%.
We interface these methods directly with the data storage and provenance technologies at the National Synchtrotron Light Source II, thus demonstrating the potential for RL to increase the scientific output of beamlines, and layout the framework for how to achieve this impact


System Requirements
===================


Hardware Requirements
*********************
While this work can be reproduced using the CPU for reinforcement learning agent training,
it is strongly recommended to use a suitable CUDA enabled GPU for the training.

Software Requirements
*********************

OS Requirements
---------------
This package has been tested exclusively on Linux operating systems containing CUDA enabled GPUs.

- Ubuntu 18.04
- PopOS 20.04

Python dependencies
-------------------

This package mainly depends on the ``tensorforce`` RL stack and some
scientific utilities::

    tensorforce
    tensorboard
    tensorflow
    numpy
    matplotlib
    pandas

The version of tensorflow/tensorforce used only has wheels for py35-py38

Getting Started
===============

Installation guide
******************


Install from github::

    $ python3 -m venv bs_env
    $ source bs_env/bin/activate
    $ git clone https://github.com/bnl/pub-Maffettone_2021_02
    $ cd pub-Maffettone_2021_02
    $ python -m pip install --upgrade pip wheel
    $ python -m pip install .

A simple demonstration
**********************
Example code of the training pipeline used in  the study is available in the `examples module <./bad_seeds/examples/>`_.

One example is given to reproduce the data for comparing the impact of batch size on learning curves::

    $ python bad_seeds/examples/variable_batch_size.py
    batch_64: SetupArgs(batch_size=64, env_version=2, out_path=PosixPath('./bad_seeds/examples/example_results'), num_episodes=250)
    X Physical GPUs, 1 Logical GPU
    Episodes: 100%|█| 250/250 [02:17, reward=83.33, ts/ep=16, sec/ep=0.27, ms/ts=16.
    batch_512: SetupArgs(batch_size=512, env_version=2, out_path=PosixPath('./bad_seeds/examples/example_results'), num_episodes=250)
    X Physical GPUs, 1 Logical GPU
    Episodes: 100%|█| 250/250 [02:20, reward=60.87, ts/ep=33, sec/ep=0.54, ms/ts=16.

A second example is given for comparing the impact of a variable time limit on the measurement episodes.::

    $ python bad_seeds/examples/variable_time_limit.py
    20_default_16: SetupArgs(time_limit=20, batch_size=16, env_version=2, out_path=PosixPath('./bad_seeds/examples/example_results'), num_episodes=250)
    X Physical GPUs, 1 Logical GPU
    Episodes: 100%|█| 250/250 [01:29, reward=100.00, ts/ep=20, sec/ep=0.34, ms/ts=16
    40_default_16: SetupArgs(time_limit=40, batch_size=16, env_version=2, out_path=PosixPath('./bad_seeds/examples/example_results'), num_episodes=250)
    X Physical GPUs, 1 Logical GPU
    Episodes: 100%|█| 250/250 [02:51, reward=100.00, ts/ep=40, sec/ep=0.66, ms/ts=16
    70_default_16: SetupArgs(time_limit=70, batch_size=16, env_version=2, out_path=PosixPath('./bad_seeds/examples/example_results'), num_episodes=250)
    X Physical GPUs, 1 Logical GPU
    Episodes: 100%|█| 250/250 [04:56, reward=100.00, ts/ep=70, sec/ep=1.16, ms/ts=16

In both cases the default is to do a ``simple`` experiment for the sake of demonstration.
If ``simple=False`` is set in the main, a more complete (and more expensive!) experiment will be run which will converge
to similiar results presented in the `published data folder <./published_results>`_.

Matplotlib functions to cast this data as seen in the paper are presented in the `plot module <./bad_seeds/plot/gen_figs.py>`_.

To generate Figures 7 and 8 from the paper ::

   $ python bad_seeds/examples/paper_figures.py  --show
   wrote /home/tcaswell/source/bnl/nsls-ii/bad-seeds/figure_7.png
   wrote /home/tcaswell/source/bnl/nsls-ii/bad-seeds/figure_8.png

or their equivalents on locally generated results ::

   $ python bad_seeds/examples/example_figures.py --show
   wrote /home/tcaswell/source/bnl/nsls-ii/bad-seeds/ideal_training.png
   wrote /home/tcaswell/source/bnl/nsls-ii/bad-seeds/time_constrained_16.png
