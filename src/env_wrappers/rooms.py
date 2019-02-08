import numpy as np
from gym import Wrapper

class Rooms(Wrapper):
    def __init__(self, env, args):
        super(Rooms, self).__init__(env)
        self.gamma = float(args['--gamma'])
        self.rNotTerm = -1 + (self.gamma - 1) * float(args['--initq'])
        self.rTerm = 0 - float(args['--initq'])

    def get_g(self):
        x = np.random.randint(self.env.unwrapped.nR)
        y = np.random.randint(self.env.unwrapped.nC)
        return np.array(self.env.unwrapped.rescale([x,y]))

    def get_r(self, s, g):
        term = np.linalg.norm(s-g, axis=-1) < 0.001
        reward = term * self.rTerm + (1 - term) * self.rNotTerm
        return term, reward

    def get_stats(self):
        stats = {}
        return stats

    @property
    def state_dim(self):
        return 2,

    @property
    def goal_dim(self):
        return 2,

    @property
    def action_dim(self):
        return 4
