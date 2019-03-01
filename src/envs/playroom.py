import numpy as np
from gym import Env
from random import randint

class Actions:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    TOUCH = 4


class Obj():
    def __init__(self, env, dep):
        self.env = env
        self.dep = dep
        self.init()

    def init(self):
        self.s = 0

    @property
    def state(self):
        return self.s

    @property
    def high(self):
        return [len(self.dep)]

    @property
    def low(self):
        return [0]

def genDep():
    seen = set()
    x, y = randint(0, 14), randint(0, 14)
    while True:
        seen.add((x, y))
        yield (x, y)
        x, y = randint(0, 14), randint(0, 14)
        while (x, y) in seen:
            x, y = randint(0, 14), randint(0, 14)

POS_LIST = [(9, 2), (2, 14), (7, 10), (8, 1), (2, 0), (4, 7), (14, 3), (9, 9), (0, 2), (4, 5), (1, 0), (14, 1), (7, 3), (2, 1), (12, 5), (14, 1), (14, 8), (5, 5), (1, 13), (7, 14), (2, 3), (8, 0), (5, 12), (11, 6), (14, 10), (4, 10), (8, 11), (4, 1), (3, 7), (6, 14)]


class Playroom(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, multistart=True):
        self.desc = np.asarray([
            " _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ",
            "|                             |",
            "|                             |",
            "|                             |",
            "|                             |",
            "|                             |",
            "|                             |",
            "|                             |",
            "|                             |",
            "|                             |",
            "|                             |",
            "|                             |",
            "|                             |",
            "|                             |",
            "|                             |",
            "|_ _ _ _ _ _ _ _ _ _ _ _ _ _ _|",
        ], dtype='c')
        self.nR = self.desc.shape[0]-1
        self.nC = (self.desc.shape[1]-1)/2
        self.multistart = multistart
        self.N = 10
        self.L = 3
        self.objectDepGen = self.genDep()
        self.objects = []
        for i in range(self.N):
            self.objects.append(Obj(self,
                                    dep=[POS_LIST[self.L * i + j] for j in range(self.L)]))
        self.initialize()

    def genDep(self):
        seen = set()
        x, y = randint(0, self.nR), randint(0, self.nC)
        while True:
            seen.add((x, y))
            yield (x, y)
            x, y = randint(0, self.nR), randint(0, self.nC)
            while (x, y) in seen:
                x, y = randint(0, self.nR), randint(0, self.nC)

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

    def step(self, a, tutor=False):
        env_a = a
        if self.lastaction is not None and np.random.rand() < 0.2:
            env_a = self.lastaction
        self.lastaction = a

        if env_a == Actions.UP and self.desc[1 + self.x, 1 + 2 * self.y] == b" ":
            self.x = min(self.x + 1, self.nR - 1)

        if env_a == Actions.DOWN and self.desc[self.x, 1 + 2 * self.y] == b" ":
            self.x = max(self.x - 1, 0)

        if env_a == Actions.RIGHT and self.desc[1 + self.x, 2 + 2 * self.y] == b" ":
            self.y = min(self.y + 1, self.nC - 1)

        if env_a == Actions.LEFT and self.desc[1 + self.x, 2 * self.y] == b" ":
            self.y = max(self.y - 1, 0)

        elif env_a == Actions.TOUCH:
            for obj in self.objects:
                if obj.s < obj.high[0] and (self.x, self.y) == obj.dep[obj.s]:
                    obj.s = np.random.choice([obj.s, obj.s + 1], p=[0, 1])

        return np.array(self.state), 0, 0, {}

    def reset(self):
        self.initialize()
        return np.array(self.state)

    @property
    def high(self):
        res = [self.nC - 1, self.nR - 1]
        for obj in self.objects:
            res += obj.high
        return res

    def rescale(self, state):
        return [(a - c) / (b - c) for a, b, c in zip(state, self.high, self.low)]

    def unscale(self, state):
        return [a *(b-c) + c for a, b, c in zip(state, self.high, self.low)]

    @property
    def state(self):
        res = [self.x, self.y] + [obj.state for obj in self.objects]
        return self.rescale(res)

    @property
    def low(self):
        res = [0, 0]
        for obj in self.objects:
            res += obj.low
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

    def touch(self, x, y):
        a = self.go(x, y)
        if a is None:
            return Actions.TOUCH, False
        else:
            return a, False

    def opt_action(self, t):
        obj = self.objects[t]
        if obj.state == obj.high:
            return -1, True
        else:
            dep = obj.dep[obj.s]
            return self.touch(dep[0], dep[1])

if __name__ == '__main__':
    env = Playroom(multistart=True)
    s = env.reset()
    task = np.random.choice([1
                             ])
    while True:
        print(s)
        a, done = env.opt_action(task)
        if done:
            break
        else:
            a = np.expand_dims(a, axis=1)
            s = env.step(a, True)[0]

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
