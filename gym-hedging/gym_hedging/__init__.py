from gym.envs.registration import register

register(
    id='hedging-v0',
    entry_point='gym_hedging.envs:HedgingEnv',
)