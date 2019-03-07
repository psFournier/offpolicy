import numpy as np
from gym import Wrapper, spaces

class PlayroomReward(Wrapper):
    def __init__(self, env, args=None):
        super(PlayroomReward, self).__init__(env)
        assert int(args['--targetClip']) == 0
        self.gamma = float(args['--gamma'])
        self.rNotTerm = -1 + (self.gamma - 1) * float(args['--initq'])
        self.rTerm = 0 - float(args['--initq'])
        self.mode = 'train'

    def reset(self, state, mode='train'):
        exp = {}
        exp['s0'] = state
        exp['goal'] = np.empty(0)
        return exp

    def make_input(self, exp):
        input = [np.expand_dims(i, axis=0) for i in [exp['s0'], exp['goal']]]
        return input

    def get_r(self, s, g, r=None, term=None):
        return term, r

    def select_goal_train(self):
        return np.empty(0)

    def process_trajectory(self, trajectory):
        new_trajectory = []
        goals = np.expand_dims(trajectory[-1]['goal'], axis=0)
        for i, exp in enumerate(reversed(trajectory)):
            if i == 0:
                exp['next'] = None
            else:
                exp['next'] = trajectory[-i]
            exp['goal'] = goals
            exp['terminal'] = np.expand_dims(exp['terminal'], axis=0)
            exp['reward'] = np.expand_dims(exp['reward'], axis=0)
            new_trajectory.append(exp)
        return new_trajectory

    @property
    def state_dim(self):
        return 27,

    @property
    def goal_dim(self):
        return 0,

    @property
    def action_dim(self):
        return 5
