import numpy as np
from gym import Wrapper

class Rooms(Wrapper):
    def __init__(self, env, args):
        super(Rooms, self).__init__(env)
        self.gamma = float(args['--gamma'])
        self.multigoal = bool(int(args['--multigoal']))
        self.rNotTerm = -1 + (self.gamma - 1) * float(args['--initq'])
        self.rTerm = 0 - float(args['--initq'])

    def reset(self, state):
        exp = {}
        exp['s0'] = state

        if self.multigoal:
            x = np.random.randint(self.env.unwrapped.nR)
            y = np.random.randint(self.env.unwrapped.nC)
        else:
            x = self.env.unwrapped.nR - 1
            y = self.env.unwrapped.nC - 1
        exp['goal'] = np.array(self.env.unwrapped.rescale([x,y]))

        return exp

    def make_input(self, exp):
        input = [np.expand_dims(i, axis=0) for i in [exp['s0'], exp['goal']]]
        return input

    def get_r(self, exp, r=None, term=None):
        s, g = exp['s1'], exp['goal']
        exp['terminal'] = np.linalg.norm(s-g, axis=-1) < 0.001
        exp['reward'] = exp['terminal'] * self.rTerm + (1 - exp['terminal']) * self.rNotTerm
        return exp

    @property
    def state_dim(self):
        return 2,

    @property
    def goal_dim(self):
        return 2,

    @property
    def action_dim(self):
        return 4
