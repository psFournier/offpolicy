import numpy as np
from gym import Wrapper

class Playroom(Wrapper):
    def __init__(self, env, args):
        super(Playroom, self).__init__(env)
        self.gamma = float(args['--gamma'])
        self.multigoal = bool(int(args['--multigoal']))
        self.rNotTerm = -1 + (self.gamma - 1) * float(args['--initq'])
        self.rTerm = 0 - float(args['--initq'])

    def reset(self, state):
        exp = {}
        exp['s0'] = state

        v = np.zeros(self.state_dim)
        idx = np.random.randint(self.state_dim[0])
        v[idx] = 1

        g = np.zeros(self.state_dim)
        if self.multigoal:
            l, h = self.env.unwrapped.low[idx], self.env.unwrapped.high[idx]
            g[idx] = (np.random.randint(l, h) - l) / (h -l)
        else:
            g[idx] = 1
        exp['goal'] = np.hstack([g, v])

        return exp

    def make_input(self, exp):
        input = [np.expand_dims(i, axis=0) for i in [exp['s0'], exp['goal']]]
        return input

    def get_r(self, exp, r=None, term=None):
        s, g = exp['s1'], exp['goal']
        g, v = np.split(g, self.state_dim, axis=-1)
        exp['terminal'] = np.linalg.norm(np.multiply(v, s-g), axis=-1) < 0.001
        exp['reward'] = exp['terminal'] * self.rTerm + (1 - exp['terminal']) * self.rNotTerm
        return exp

    def process_trajectory(self, trajectory, base_util=None, hindsight=True):
        if base_util is None:
            u = np.zeros(shape=(self.N,))
        else:
            u = base_util
        u = np.expand_dims(u, axis=1)
        # mcr = np.zeros(shape=(self.N,))
        for exp in reversed(trajectory):
            u = self.gamma * u
            if hindsight:
                u[np.where(exp['r1'] > exp['r0'])] = 1

            # u_idx = np.where(u != 0)
            # mcr[u_idx] = exp['r1'][u_idx] + self.gamma * mcr[u_idx]
            exp['u'] = u.squeeze()
            # exp['mcr'] = mcr
            if any(u!=0):
                self.buffer.append(exp.copy())

    @property
    def state_dim(self):
        return 2+len(self.env.objects),

    @property
    def goal_dim(self):
        return 2*self.state_dim[0],

    @property
    def action_dim(self):
        return 5
