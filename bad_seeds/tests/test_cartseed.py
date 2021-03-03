from bad_seeds.environments.cartseed import CartSeed, CartSeedCountdown


# Initialization
# Reset
# bad initialization
# play
# sequential
# frozen order
# revisiting on
# revisiting off
# too many bad seeds
# variable number bad seeds
# terminal exit
# Not callable as reward
# Multi functional max_episode_timesteps (with and without kwarg)
# Varying levels of max_episode_timesteps


def test_env_start(default_cartseed):
    """
    Test initialization
    Test reset state
    """
    state = default_cartseed.reset()
    print(f"Start state: {state}")
    print(f"Environmental snaphot:\n {default_cartseed.seeds}")
    print(f"Number of bad seeds: {default_cartseed.bad_seed_count}")
    print(f"Max timesteps {default_cartseed.max_episode_timesteps()}")
    assert True


def test_variable_env_start(default_cartseedcountdown):
    """
    Test initialization
    Test reset state
    """
    pass


def test_invalid_init():
    """Make bad environments and collect raise or failures"""
    pass


def test_max_episodes_timesteps():
    """Initialize a handful of environments and assure their max_episodes_timesteps is consistent"""
    pass


def test_revisiting():
    """Exhaust and reset an environment with revisiting on vs off"""
    pass
