import numpy as np
from gym import Wrapper

class Playroom(Wrapper):
    def __init__(self, env, args):
        super(Playroom, self).__init__(env)
        self.gamma = float(args['--gamma'])
        self.rNotTerm = -1 + (self.gamma - 1) * float(args['--initq'])
        self.rTerm = 0 - float(args['--initq'])
        self.her = float(args['--her']) * int(args['--ep_steps'])
        self.N = len(self.env.objects)

    def reset(self, state):
        exp = {}
        exp['s0'] = state
        exp['rParams'] = self.get_rParams()
        return exp

    def get_rParams(self):
        theta1 = np.random.uniform(size=self.N)
        theta1 /= sum(theta1)
        theta2 = np.random.randint(1, self.env.L + 1, size=self.N) / self.env.L
        return np.hstack([theta1, theta2])

    def get_r(self, s, rParams):
        w, g = np.split(rParams, 2, axis=-1)
        pos, objs = np.split(s, [2], axis=-1)
        d = np.linalg.norm(np.multiply(w, objs-g), axis=-1)
        t = d < 0.001
        r = t * self.rTerm + (1 - t) * self.rNotTerm
        return r, t

    def process_trajectory(self, trajectory):
        l = len(trajectory)
        utilities = np.zeros(self.N)
        processed = []

        for i, exp in enumerate(trajectory):
            if i == 0:
                exp['prev'] = None
            else:
                exp['prev'] = trajectory[i - 1]
            if i == l-1:
                exp['next'] = None
            else:
                exp['next'] = trajectory[i+1]

        for exp in reversed(trajectory):
            changes = np.where(exp['s0'][2:] != exp['s1'][2:])[0]
            utilities *= self.gamma
            utilities[changes] += 1
            exp['u'] = utilities
            processed.append(exp)

        return processed

    # def process_trajectory(self, trajectory):
    #     rParams = np.expand_dims(trajectory[-1]['rParams'], axis=0)
    #     new_trajectory = []
    #     n_changes = 0

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

        # for i, exp in enumerate(reversed(trajectory)):
        #     if i == 0:
        #         exp['next'] = None
        #     else:
        #         exp['next'] = trajectory[-i]
        #
        #     # if self.her != 0:
        #     #     virtual_goals = [np.hstack([trajectory[idx]['s1'], self.vs[c]]) for idx, c in virtual_idx if idx >= i]
        #     #     exp['goal'] = np.vstack([trajectory[-1]['goal']] + virtual_goals)
        #     # else:
        #     #     exp['goal'] = np.expand_dims(trajectory[-1]['goal'], axis=0)
        #
        #     # Reservoir sampling for HER
        #     if self.her != 0:
        #         changes = np.where(exp['s0'][2:] != exp['s1'][2:])[0]
        #         for change in changes:
        #             n_changes += 1
        #             v = self.vs[change]
        #             if goals.shape[0] <= self.her:
        #                 goals = np.vstack([goals, np.hstack([exp['s1'], v])])
        #             else:
        #                 j = np.random.randint(1, n_changes + 1)
        #                 if j <= self.her:
        #                     goals[j] = np.hstack([exp['s1'], v])
        #
        #     exp['rParams'] = rParams
        #     # exp['reward'], exp['terminal'] = self.get_r(exp['s1'], exp['rParams'])
        #     new_trajectory.append(exp)
        #
        # return new_trajectory

    @property
    def state_dim(self):
        return 2 + self.N,

    @property
    def goal_dim(self):
        return 2 * self.N,

    @property
    def action_dim(self):
        return 5
