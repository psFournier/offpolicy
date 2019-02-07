import numpy as np
from gym import Wrapper

class Rooms(Wrapper):
    def __init__(self, env, args):
        super(Rooms, self).__init__(env)
        self.her_p = float(args['--her_p'])
        self.gamma = float(args['--gamma'])
        self.rNotTerm = -1 + (self.gamma - 1) * float(args['--initq'])
        self.rTerm = 0 - float(args['--initq'])

    def get_g(self):
        x = np.random.randint(self.env.unwrapped.nR)
        y = np.random.randint(self.env.unwrapped.nC)
        return np.array(self.env.unwrapped.rescale([x,y]))

    def get_r(self, s, g):
        term = np.linalg.norm(s-g) < 0.001
        reward = term * self.rTerm + (1 - term) * self.rNotTerm
        return term, reward

    def process_trajectory(self, trajectory):
        virtual_g = []
        res = []
        for exp in reversed(trajectory):
            if exp['o'] == 0:
                res.append(exp.copy())
            for vg in virtual_g:
                exp['g'] = vg
                exp['t'], exp['r1'] = self.get_r(exp['s1'], vg)
                res.append(exp.copy())
            if np.random.rand() < self.her_p:
                virtual_g.append(exp['s0'])
        return res

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
