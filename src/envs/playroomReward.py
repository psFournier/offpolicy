import numpy as np
from gym import Env
from random import randint

class Actions:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    # TOUCH = 4


class Obj():
    def __init__(self, env, dep, rewards):
        self.env = env
        self.dep = dep
        self.rewards = rewards
        self.init()

    def init(self):
        self.s = 0

    @property
    def high(self):
        return len(self.dep)

    @property
    def low(self):
        return 0

def genDep():
    seen = set()
    x, y = randint(0, 14), randint(0, 14)
    while True:
        seen.add((x, y))
        yield (x, y)
        x, y = randint(0, 14), randint(0, 14)
        while (x, y) in seen:
            x, y = randint(0, 14), randint(0, 14)

POS_LIST = [(3, 4), (5, 2), (5, 7), (4, 2), (5, 0), (4, 7), (7, 6), (0, 9), (7, 5), (4, 1), (0, 3), (9, 2), (9, 5), (8, 9), (1, 8), (1, 2), (6, 3), (5, 9), (2, 2), (7, 9), (9, 0), (4, 4), (2, 9), (1, 5), (0, 4), (6, 1), (2, 1), (2, 3), (4, 3), (2, 6), (3, 2), (1, 6), (8, 2), (7, 0), (5, 5), (0, 1), (0, 8), (2, 7), (1, 1), (5, 4), (6, 2), (3, 6), (9, 9), (0, 0), (8, 6), (6, 0), (5, 1), (4, 9), (0, 7), (7, 4), (8, 3), (8, 8), (0, 6), (5, 3), (6, 4), (1, 0), (3, 7), (1, 3), (7, 7), (4, 0), (4, 5), (6, 8), (6, 5), (4, 8), (6, 7), (9, 8), (2, 4), (8, 0), (5, 8), (7, 8), (9, 1), (3, 8), (8, 5), (3, 3), (3, 0), (6, 9), (0, 2), (9, 3), (3, 5), (2, 5), (9, 4), (8, 1), (5, 6), (3, 9), (7, 3), (0, 5), (6, 6), (1, 4), (8, 4), (2, 0), (9, 6), (2, 8), (1, 9), (1, 7), (3, 1), (4, 6), (7, 1), (7, 2), (9, 7), (8, 7)]

REWARDS = [-1.0, -1.0, -0.9595105550104429, -0.246828342467586, -1.0, -0.4975851212712543, -1.0, -0.01427247457681835, -1.0, -0.26753514590804295, -0.7506189947620457, -0.4444090015775241, -0.19717538989061323, -0.963027359529741, -0.6276921918141077, -0.907109476192222, -0.5866275910407244, 0.5300592862211041, -1.0, -1.0, 0.20562787885319178, -0.29545574152722753, 0.1128356605162989, 0.3460536387169181, 0.22510418049519232, 0.16864618843396972, -0.8014119821040651, -0.07396004801622091, -1.0, -0.18294620796604738, -0.4216855603319981, -1.0, -0.9949133580186359, -0.9440263755177624, -0.7236409618178383, -0.5286642544655418, -0.11443211461971106, -0.44008864109539253, -0.32680096327319436, 0.31556838669892173, 0.3077793373784332, -0.027073455923367393, -0.5624487013110808, 0.8891619822919219, -0.23973181258873338, -0.2048564276201315, -0.3541714440149005, -1.0, -0.3654382094872508, -0.2651482211671625, -0.35849161762597237, 0.5766838395502857, -0.3116014273214018, -0.29445225524499274, 0.161816802529538, 0.16102188003520607, 0.7320716884424446, 0.6238645059202044, 0.6775549690174666, 0.7162460484415573, 0.44589166285647636, 0.36150272400319594, 0.3168817820406526, -0.2190093281298352, -0.07371579272815026, -0.6385876216566189, 1.0, -0.12598357176525599, 0.5561403691225584, 0.24808541613711382, 0.2835762111779666, 0.25701009008150716, 0.8727119134704036, 0.08661107188365436, 1.0, 0.07845147472995473, 0.45568219948438987, 0.8688394537000326, 0.10540467739085146, 0.8801136811364318, 1.0, 0.2918216198630032, 0.9561016567731845, 0.4351692090850843, 0.919834571813411, 0.7554364270559094, -0.4619843202196676, -0.5817701550020316, 0.802877811642071, 1.0, 1.0, 0.06685698112808425, 0.3180173292602373, 1.0, 0.5352334944105231, 1.0, 1.0, 0.06299426232701977, 1.0, 1.0]

class PlayroomReward(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, multistart=True, sparsity=0):
        self.desc = np.asarray([
            " _ _ _ _ _ _ _ _ _ _ ",
            "|                   |",
            "|                   |",
            "|                   |",
            "|                   |",
            "|                   |",
            "|                   |",
            "|                   |",
            "|                   |",
            "|                   |",
            "|_ _ _ _ _ _ _ _ _ _|"
        ], dtype='c')
        self.nR = self.desc.shape[0]-1
        self.nC = (self.desc.shape[1]-1)/2
        self.multistart = multistart
        self.N = 5
        self.L = 20
        self.objects = []
        for i in range(self.N):
            self.objects.append(Obj(self,
                                    dep=[POS_LIST[(100//self.N) * i + j] for j in range(self.L)],
                                    rewards=[REWARDS[(100//self.N) * i + j] for j in range(self.L)]))
        self.sparsity = sparsity
        self.initialize()

    # def genDep(self):
    #     seen = set()
    #     x, y = randint(0, self.nR), randint(0, self.nC)
    #     while True:
    #         seen.add((x, y))
    #         yield (x, y)
    #         x, y = randint(0, self.nR), randint(0, self.nC)
    #         while (x, y) in seen:
    #             x, y = randint(0, self.nR), randint(0, self.nC)

    def initialize(self):
        if self.multistart:
            self.x = np.random.randint(self.nR)
            self.y = np.random.randint(self.nC)
        else:
            self.x = 0
            self.y = 0
        for obj in self.objects:
            obj.s = 0
        self.lastaction = None

    def step(self, a):
        env_a = a
        if self.lastaction is not None and np.random.rand() < 0:
            env_a = self.lastaction
        self.lastaction = a
        r = 0
        if env_a == Actions.UP and self.desc[1 + self.x, 1 + 2 * self.y] == b" ":
            self.x = min(self.x + 1, self.nR - 1)

        if env_a == Actions.DOWN and self.desc[self.x, 1 + 2 * self.y] == b" ":
            self.x = max(self.x - 1, 0)

        if env_a == Actions.RIGHT and self.desc[1 + self.x, 2 + 2 * self.y] == b" ":
            self.y = min(self.y + 1, self.nC - 1)

        if env_a == Actions.LEFT and self.desc[1 + self.x, 2 * self.y] == b" ":
            self.y = max(self.y - 1, 0)

        # elif env_a == Actions.TOUCH:
        for obj in self.objects:
            if (self.x, self.y) == obj.dep[obj.s] and obj.s < obj.high:
                obj.s += 1
                if obj.s % self.sparsity == 0:
                    r += self.sparsity*obj.rewards[obj.s - 1]

        return np.array(self.state), r, 0, {}

    def reset(self):
        self.initialize()
        return np.array(self.state)

    @property
    def high(self):
        res = [self.nC - 1, self.nR - 1] + [obj.high for obj in self.objects]
        return res

    def rescale(self, state):
        return [(a - c) / (b - c) for a, b, c in zip(state, self.high, self.low)]

    def unscale(self, state):
        return [a * (b - c) + c for a, b, c in zip(state, self.high, self.low)]

    @property
    def state(self):
        res = [self.x, self.y] + [obj.s for obj in self.objects]
        return self.rescale(res)

    @property
    def low(self):
        res = [0, 0] + [obj.low for obj in self.objects]
        return res

    def go(self, x, y):
        dx = x - self.x
        dy = y - self.y
        p = []
        if dx > 0:
            p.append(Actions.UP)
        elif dx < 0:
            p.append(Actions.DOWN)
        if dy > 0:
            p.append(Actions.RIGHT)
        elif dy < 0:
            p.append(Actions.LEFT)
        if p:
            return np.random.choice(p)
        else:
            return None

    # def touch(self, x, y):
    #     a = self.go(x, y)
    #     if a is None:
    #         return Actions.TOUCH, False
    #     else:
    #         return a, False

    def opt_action(self, t):
        obj = self.objects[t]
        if obj.s == obj.high:
            return -1, True
        else:
            dep = obj.dep[obj.s]
            return self.go(dep[0], dep[1]), False

if __name__ == '__main__':
    env = PlayroomReward(multistart=True, sparsity=10)
    for task in range(25):
        s = env.reset()
        sumr = 0
        while True:
            a, done = env.opt_action(task)
            if done:
                break
            else:
                a = np.expand_dims(a, axis=1)
                s, r, t, _ = env.step(a)
                sumr += r
        print(sumr)
    # s = env.reset()
    # step = 0
    # ep_step = 0
    # # task = 0
    # prop = 0.99
    # sumr = 0
    # while step < 50000:
    #     # done = (env.objects[task].s == env.objects[task].high)
    #     if ep_step >= 200:
    #         print(sumr)
    #         s = env.reset()
    #         ep_step = 0
    #         # task = np.random.choice([0, 1], p=[prop, 1 - prop])
    #     else:
    #         # a, _ = env.opt_action(task)
    #         a = np.random.randint(5)
    #         a = np.expand_dims(a, axis=1)
    #         s, r, _, _ = env.step(a)
    #         sumr += np.abs(r)
    #         ep_step += 1
    #         step += 1
    # def render(self, mode='human'):
    #     outfile = StringIO() if mode == 'ansi' else sys.stdout
    #
    #     out = self.desc.copy().tolist()
    #     out = [[c.decode('utf-8') for c in line] for line in out]
    #     taxirow, taxicol, passidx = self.decode(self.s)
    #     def ul(x): return "_" if x == " " else x
    #     if passidx < 4:
    #         out[1+taxirow][2*taxicol+1] = utils.colorize(out[1+taxirow][2*taxicol+1], 'yellow', highlight=True)
    #         pi, pj = self.locs[passidx]
    #         out[1+pi][2*pj+1] = utils.colorize(out[1+pi][2*pj+1], 'blue', bold=True)
    #     else: # passenger in taxi
    #         out[1+taxirow][2*taxicol+1] = utils.colorize(ul(out[1+taxirow][2*taxicol+1]), 'green', highlight=True)
    #
    #     # No need to return anything for human
    #     if mode != 'human':
    #         return outfile
