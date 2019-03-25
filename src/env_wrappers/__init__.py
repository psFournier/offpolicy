from env_wrappers.registration import registry, register, make

register(
    id='Rooms1-v0',
    entry_point='envs:Rooms1',
    wrapper_entry_point='env_wrappers.rooms:Rooms'
)

register(
    id='Rooms9Multi-v0',
    entry_point='envs:Rooms9',
    kwargs={'multistart': True},
    wrapper_entry_point='env_wrappers.rooms:Rooms'
)

register(
    id='Rooms9Mono-v0',
    entry_point='envs:Rooms9',
    kwargs={'multistart': False},
    wrapper_entry_point='env_wrappers.rooms:Rooms'
)

register(
    id='CartPole-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    wrapper_entry_point='env_wrappers.base:Base'
)

register(
    id='Playroom-v0',
    entry_point='envs:Playroom',
    wrapper_entry_point='env_wrappers.playroom:Playroom'
)

register(
    id='Breakout-v0',
    entry_point='gym.envs.atari:AtariEnv',
    kwargs={'game': 'breakout', 'obs_type': 'ram', 'repeat_action_probability': 0.25},
    wrapper_entry_point='env_wrappers.base:Base'
)

for s in [1, 2, 4]:
    register(
        id='PlayroomRewardSparse{}-v0'.format(s),
        entry_point='envs:PlayroomReward',
        kwargs={'sparsity': s},
        wrapper_entry_point='env_wrappers.playroomReward:PlayroomReward'
    )

register(
    id='MiniBreakout-v0',
    entry_point='envs:MiniBreakout',
    wrapper_entry_point='env_wrappers.miniBreakout:MiniBreakout'
)

# register(
#     id='Rooms2-v0',
#     entry_point='envs:Rooms2',
#     wrapper_entry_point='env_wrappers.rooms2:Rooms2'
# )