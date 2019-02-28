import numpy as np
from gym import Wrapper

class Rooms(Wrapper):
    def __init__(self, env, args):
        super(Rooms, self).__init__(env)
        self.gamma = float(args['--gamma'])
        self.rNotTerm = -1 + (self.gamma - 1) * float(args['--initq'])
        self.rTerm = 0 - float(args['--initq'])
        self.her = float(args['--her']) * int(args['--ep_steps'])
        self.mode = 'train'

    def reset(self, state):
        exp = {}
        exp['s0'] = state

        if self.mode == 'train':
            x = 0
            y = np.random.randint(self.env.unwrapped.nC)
        else:
            x = self.env.unwrapped.nR - 1
            y = np.random.randint(self.env.unwrapped.nC)

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

    def process_trajectory(self, trajectory):
        goals = np.expand_dims(trajectory[-1]['goal'], axis=0)
        new_trajectory = []
        n_changes = 0

        for i, exp in enumerate(reversed(trajectory)):
            if i == 0:
                exp['next'] = None
            else:
                exp['next'] = trajectory[-i]

            # Reservoir sampling for HER
            if self.her != 0:
                changes = np.where(exp['s0'][2:] != exp['s1'][2:])[0]
                for change in changes:
                    n_changes += 1
                    if goals.shape[0] <= self.her:
                        goals = np.vstack([goals, exp['s1']])
                    else:
                        j = np.random.randint(1, n_changes + 1)
                        if j <= self.her:
                            goals[j] = exp['s1']
            exp['goal'] = goals

            # if np.random.rand() < self.her:
            #     goals = np.vstack([goals, exp['s1']])
            # exp['goal'] = goals

            exp = self.get_r(exp)
            new_trajectory.append(exp)
        return new_trajectory

    @property
    def state_dim(self):
        return 2,

    @property
    def goal_dim(self):
        return 2,

    @property
    def action_dim(self):
        return 4
