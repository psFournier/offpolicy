import numpy as np
from gym import Wrapper, spaces

class Base(Wrapper):
    def __init__(self, env, args):
        super(Base, self).__init__(env)
        self.gamma = float(args['--gamma'])
        self.multigoal = bool(int(args['--multigoal']))
        self.rNotTerm = -1 + (self.gamma - 1) * float(args['--initq'])
        self.rTerm = 0 - float(args['--initq'])

    def get_g(self):
        return np.empty(0)

    def get_r(self, s, g, r=None, term=None):
        assert g.size != 0 or r is not None
        if g.size != 0:
            term = np.linalg.norm(s - g, axis=-1) < 0.001
            r = term * self.rTerm + (1 - term) * self.rNotTerm
        return term, r

    def get_stats(self):
        stats = {}
        return stats

    @property
    def state_dim(self):
        return self.env.observation_space.low.shape[0],

    @property
    def goal_dim(self):
        return 0,

    @property
    def action_dim(self):
        return self.env.action_space.n
