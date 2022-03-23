from gym.envs.registration import registry, register, make, spec

register(
    id='Quad2D-v0',
    entry_point='gym_quad.envs:Quad2DEnv',
    # max_episode_steps = 10000,
)

