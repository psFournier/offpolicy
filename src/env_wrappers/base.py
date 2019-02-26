import numpy as np
from gym import Wrapper, spaces

class Base(Wrapper):
    def __init__(self, env, args=None):
        super(Base, self).__init__(env)
        assert int(args['--targetClip']) == 0
        self.gamma = float(args['--gamma'])
        self.multigoal = bool(int(args['--multigoal']))
        self.rNotTerm = -1 + (self.gamma - 1) * float(args['--initq'])
        self.rTerm = 0 - float(args['--initq'])

    def reset(self, state):
        exp = {}
        exp['s0'] = state
        exp['goal'] = np.empty(0)
        return exp

    def make_input(self, exp):
        input = [np.expand_dims(i, axis=0) for i in [exp['s0'], exp['goal']]]
        return input

    def get_r(self, exp, r=None, term=None):
        exp['terminal'] = term
        exp['reward'] = r
        return exp

    @property
    def state_dim(self):
        return self.env.observation_space.low.shape[0],

    @property
    def goal_dim(self):
        return 0,

    @property
    def action_dim(self):
        return self.env.action_space.n
