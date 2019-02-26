from env_wrappers.registration import registry, register, make

register(
    id='Rooms1-v0',
    entry_point='envs:Rooms1',
    wrapper_entry_point='env_wrappers.rooms:Rooms'
)

register(
    id='Rooms9-v0',
    entry_point='envs:Rooms9',
    wrapper_entry_point='env_wrappers.rooms:Rooms'
)

register(
    id='CartPole-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    wrapper_entry_point='env_wrappers.base:Base'
)

# register(
#     id='Rooms2-v0',
#     entry_point='envs:Rooms2',
#     wrapper_entry_point='env_wrappers.rooms2:Rooms2'
# )