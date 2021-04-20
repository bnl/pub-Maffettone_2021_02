"""
An example to produce the trained model used in the bluesky-adaptive tutorial:
https://github.com/bluesky/tutorial-adaptive
"""

from bad_seeds.training.train_a2c_cartseed import set_up
from pathlib import Path
from tensorforce.execution import Runner


def main(
        time_limit=None,
        scoring="default",
        batch_size=16,
        gpu_idx=0,
        env_version=1,
        seed_count=9,
        max_count=10,
        out_path=None,
        num_episodes=int(3 * 10 ** 3),
):
    env, agent = set_up(
        time_limit=time_limit,
        scoring=scoring,
        batch_size=batch_size,
        gpu_idx=gpu_idx,
        env_version=env_version,
        seed_count=seed_count,
        max_count=max_count,
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


def load_agent(
        time_limit=None,
        scoring="default",
        batch_size=16,
        gpu_idx=0,
        env_version=1,
        seed_count=9,
        max_count=10,
        out_path=None,
):
    env, agent = set_up(
        time_limit=time_limit,
        scoring=scoring,
        batch_size=batch_size,
        gpu_idx=gpu_idx,
        env_version=env_version,
        seed_count=seed_count,
        max_count=max_count,
        out_path=out_path,
    )
    if out_path is None:
        out_path = Path()
    else:
        out_path = Path(out_path).expanduser()
    agent.restore(directory=str(out_path / "saved_models"))
    runner = Runner(agent=agent, environment=env)
    runner.run(num_episodes=20)
    return agent


if __name__ == "__main__":
    main(out_path="./bluesky-tutorial-time50",
         time_limit=50,
         num_episodes=1_000)
    # agent = load_agent()
