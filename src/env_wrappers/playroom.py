import numpy as np
from gym import Wrapper

class Playroom(Wrapper):
    def __init__(self, env, args):
        super(Playroom, self).__init__(env)
        self.gamma = float(args['--gamma'])
        self.rNotTerm = -1 + (self.gamma - 1) * float(args['--initq'])
        self.rTerm = 0 - float(args['--initq'])
        self.her = float(args['--her']) * int(args['--ep_steps'])
        vs = np.zeros(shape=(self.state_dim[0] - 2, self.state_dim[0]))
        vs[np.arange(self.state_dim[0] - 2), range(2, self.state_dim[0])] = 1
        self.vs = vs / np.sum(vs, axis=1, keepdims=True)

    def reset(self, state):
        exp = {}
        exp['s0'] = state
        exp['goal'] = self.select_goal_test()
        return exp

    def select_goal_test(self):
        v = np.zeros(self.state_dim)
        self.idx = 3
        v[self.idx] = 1
        g = np.zeros(self.state_dim)
        g[self.idx] = 1
        return np.hstack([g, v])

    def select_goal_train(self):
        return self.select_goal_test()

    def make_input(self, exp):
        input = [np.expand_dims(i, axis=0) for i in [exp['s0'], exp['goal']]]
        return input

    def get_r(self, s, g, r=None, term=None):
        g, v = np.split(g, self.state_dim, axis=-1)
        t = np.linalg.norm(np.multiply(v, s-g), axis=-1) < 0.001
        r = t * self.rTerm + (1 - t) * self.rNotTerm
        return t, r

    def process_trajectory(self, trajectory):
        goals = np.expand_dims(trajectory[-1]['goal'], axis=0)
        new_trajectory = []
        n_changes = 0

        # Reservoir sampling for HER
        # if self.her != 0:
        #     virtual_idx = []
        #     for i, exp in enumerate(reversed(trajectory)):
        #         changes = np.where(exp['s0'][2:] != exp['s1'][2:])[0]
        #         for change in changes:
        #             n_changes += 1
        #             if len(virtual_idx) < self.her:
        #                 virtual_idx.append((i, change))
        #             else:
        #                 j = np.random.randint(0, n_changes)
        #                 if j < self.her:
        #                     virtual_idx[j] = (i, change)

        for i, exp in enumerate(reversed(trajectory)):
            if i == 0:
                exp['next'] = None
            else:
                exp['next'] = trajectory[-i]

            # if self.her != 0:
            #     virtual_goals = [np.hstack([trajectory[idx]['s1'], self.vs[c]]) for idx, c in virtual_idx if idx >= i]
            #     exp['goal'] = np.vstack([trajectory[-1]['goal']] + virtual_goals)
            # else:
            #     exp['goal'] = np.expand_dims(trajectory[-1]['goal'], axis=0)

            # Reservoir sampling for HER
            if self.her != 0:
                changes = np.where(exp['s0'][2:] != exp['s1'][2:])[0]
                for change in changes:
                    n_changes += 1
                    v = self.vs[change]
                    if goals.shape[0] <= self.her:
                        goals = np.vstack([goals, np.hstack([exp['s1'], v])])
                    else:
                        j = np.random.randint(1, n_changes + 1)
                        if j <= self.her:
                            goals[j] = np.hstack([exp['s1'], v])
            exp['goal'] = goals

            exp['terminal'], exp['reward'] = self.get_r(exp['s1'], exp['goal'])
            new_trajectory.append(exp)

        return new_trajectory

    @property
    def state_dim(self):
        return 2+len(self.env.objects),

    @property
    def goal_dim(self):
        return 2*self.state_dim[0],

    @property
    def action_dim(self):
        return 5
