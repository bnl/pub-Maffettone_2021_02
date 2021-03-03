import pytest
from bad_seeds.environments.cartseed import CartSeedCountdown, CartSeed
from tensorforce.environments import Environment


@pytest.fixture
def default_cartseed():
    """
    This environment should lock into place the basics, with 10 seeds, each requiring
    Returns
    -------
    Environment
    """
    env = CartSeed(
        seed_count=10,
        bad_seed_count=3,
        frozen_order=True,
    )
    env = Environment.create(environment=env)
    return env


@pytest.fixture
def default_cartseedcountdown():
    """
    This environment should lock into place the basics, with 10 seeds, each requiring
    Returns
    -------
    Environment
    """
    env = CartSeedCountdown(
        seed_count=10,
        bad_seed_count=3,
        frozen_order=True,
    )
    env = Environment.create(environment=env)
    return env


@pytest.fixture
def default_setup():
    from bad_seeds.training.train_a2c_cartseed import set_up

    return set_up()
