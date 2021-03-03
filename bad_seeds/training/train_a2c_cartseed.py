from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner
from bad_seeds.environments.cartseed import CartSeed, CartSeedCountdown
from bad_seeds.utils.tf_utils import tensorflow_settings
from pathlib import Path


def set_up(
    time_limit=50,
    scoring=None,
    gpu_idx=0,
    batch_size=16,
    env_version=1,
    seed_count=10,
    out_path=None,
):
    """
    Set up a rushed CartSeed agent with less time than it needs to complete an episode.
    Parameters
    ----------
    time_limit : int, None
        Turn time limit for episode
    scoring : str in {'t22', 'tt5', 'monotonic', 'linear', 'square', 'default'
        Name of reward function
    gpu_idx : int
        optional index for GPU
    batch_size : int
        Batch size for training
    env_version : int in {1, 2}
        Environment version. 1 being ideal time, 2 being time limited
    seed_count : int
        Number of bad seeds
    out_path : path
        Toplevel dir for output of models and checkpoints

    Returns
    -------
    Environment
    Agent
    """

    def tt2(state, *args):
        if state[1] >= 5:
            return 2
        else:
            return 1

    def tt5(state, *args):
        if state[1] >= 5:
            return 5
        else:
            return 1

    def monotonic(state, *args):
        # This worked but would be better described as heavyside linear
        return float(state[1] > 5) * state[1]

    def linear(state, *args):
        return state[1]

    def square(state, *args):
        return state[1] ** 2

    def default(state, *args):
        return 1

    func_dict = dict(
        tt2=tt2,
        tt5=tt5,
        monotonic=monotonic,
        linear=linear,
        square=square,
        default=default,
    )

    tensorflow_settings(gpu_idx)
    if out_path is None:
        out_path = Path().absolute()
    else:
        out_path = Path(out_path).expanduser().absolute()
    if env_version == 1:
        environment = CartSeed(
            seed_count=seed_count,
            bad_seed_count=None,
            max_count=10,
            sequential=True,
            revisiting=True,
            bad_seed_reward_f=func_dict.get(scoring, None),
            measurement_time=time_limit,
        )
    elif env_version == 2:
        environment = CartSeedCountdown(
            seed_count=seed_count,
            bad_seed_count=None,
            max_count=10,
            sequential=True,
            revisiting=True,
            bad_seed_reward_f=func_dict.get(scoring, None),
            measurement_time=time_limit,
        )
    else:
        raise NotImplementedError
    env = Environment.create(environment=environment)
    agent = Agent.create(
        agent="a2c",
        batch_size=batch_size,
        environment=env,
        summarizer=dict(
            directory=out_path
            / "training_data/a2c_cartseed/{}_{}_{}_{}".format(
                env_version, time_limit, scoring, batch_size
            ),
            labels="all",
            frequency=1,
        ),
    )

    return env, agent


def manual_main():
    """
    A manual looping main that shows how each step of the reinforcement learning proceeds in a given episode.
    This is strictly instructional.

    Returns
    -------
    None
    """
    env, agent = set_up()
    for i in range(100):
        states = env.reset()
        terminal = False
        episode_reward = 0
        episode_len = 0
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = env.execute(actions=actions)
            episode_reward += reward
            episode_len += 1
            agent.observe(terminal=terminal, reward=reward)
        print(f"Episode reward: {episode_reward}. Episode length {episode_len}")


def main(
    *,
    time_limit=None,
    scoring="default",
    batch_size=16,
    gpu_idx=0,
    env_version=2,
    out_path=None,
    num_episodes=int(3 * 10 ** 3),
):
    """
    A self contained set up of the environment and run.
    Can be used to create all of the figures associated in the reference for variable batch size and
    variable time limit. All experiments use 10 'seeds'.

    Parameters
    ----------
    time_limit : int, None
        Turn time limit for episode
    scoring : str in {'t22', 'tt5', 'monotonic', 'linear', 'square', 'default'
        Name of reward function
    batch_size : int
        Batch size for training
    gpu_idx : int
        optional index for GPU
    env_version : int in {1, 2}
        Environment version. 1 being ideal time, 2 being time limited
    out_path : path
        Toplevel dir for output of models and checkpoints
    num_episodes: int
        Number of episodes to learn over

    Returns
    -------
    None

    """
    env, agent = set_up(
        time_limit=time_limit,
        scoring=scoring,
        batch_size=batch_size,
        gpu_idx=gpu_idx,
        env_version=env_version,
        out_path=out_path,
    )
    runner = Runner(agent=agent, environment=env)
    runner.run(num_episodes=num_episodes)
    if out_path is None:
        out_path = Path()
    else:
        out_path = Path(out_path).expanduser()
    agent.save(directory=str(out_path / "saved_models"))
    agent.close()
    env.close()
