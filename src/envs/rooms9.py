import numpy as np
from gym import Env
from random import randint
from six import StringIO
import sys
from gym import utils

class Actions:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Obj():
    def __init__(self, env, prop, dep, tutor_only=False):
        self.env = env
        self.prop = prop
        self.dep = dep
        self.env.objects.append(self)
        self.init()
        self.tutor_only = tutor_only

    def init(self):
        self.s = 0

    # def touch(self, tutor):
    #     if self.s == 0 and (not self.tutor_only or tutor):
    #         self.s = np.random.choice([0, 1], p=self.prop)

    @property
    def state(self):
        return [self.s]

    @property
    def high(self):
        return [len(self.dep)]

    @property
    def low(self):
        return [0]

class Rooms9(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, args):
        self.desc = np.asarray([
            " _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ",
            "|         |         |         |",
            "|         |         |         |",
            "|         |                   |",
            "|         |         |         |",
            "|_ _   _ _|_ _ _ _ _|_ _   _ _|",
            "|         |         |         |",
            "|         |         |         |",
            "|                             |",
            "|         |         |         |",
            "|_ _   _ _|_ _   _ _|_ _   _ _|",
            "|         |         |         |",
            "|         |         |         |",
            "|                   |         |",
            "|         |         |         |",
            "|_ _ _ _ _|_ _ _ _ _|_ _ _ _ _|",
        ], dtype='c')
        self.nR = self.desc.shape[0]-1
        self.nC = (self.desc.shape[1]-1)/2
        self.tutoronly = [int(f) for f in args['--tutoronly'].split(',')]
        self.initialize()


    def initialize(self):
        self.x = np.random.randint(self.nR)
        self.y = np.random.randint(self.nC)
        # self.x = 7
        # self.y = 7
        self.objects = []

        for i, o in enumerate(self.objects):
            o.tutor_only = (i+2 in self.tutoronly)

        self.initstate = self.state.copy()
        self.lastaction = None

    def step(self, a, tutor=False):
        env_a = a
        if self.lastaction is not None and np.random.rand() < 0.2:
            env_a = self.lastaction
            # print('noise')
        self.lastaction = a

        if env_a == Actions.UP and self.desc[1 + self.x, 1 + 2 * self.y] == b" ":
            self.x = min(self.x + 1, self.nR - 1)

        if env_a == Actions.DOWN and self.desc[self.x, 1 + 2 * self.y] == b" ":
            self.x = max(self.x - 1, 0)

        if env_a == Actions.RIGHT and self.desc[1 + self.x, 2 + 2 * self.y] == b" ":
            self.y = min(self.y + 1, self.nC - 1)

        if env_a == Actions.LEFT and self.desc[1 + self.x, 2 * self.y] == b" ":
            self.y = max(self.y - 1, 0)

        return np.array(self.state),

    def underagent(self):
        for i, obj in enumerate(self.objects):
            if obj.x == self.x and obj.y == self.y:
                return i+1
        return 0

    def reset(self):
        self.initialize()
        return np.array(self.state)

    def go(self, x , y):
        dx = x - self.x
        dy = y - self.y
        p = []
        if dx > 0 and self.desc[1 + self.x, 1 + 2 * self.y] == b" ":
            p.append(Actions.UP)
        elif dx < 0 and self.desc[self.x, 1 + 2 * self.y] == b" ":
            p.append(Actions.DOWN)
        if dy > 0 and self.desc[1 + self.x, 2 + 2 * self.y] == b" ":
            p.append(Actions.RIGHT)
        elif dy < 0 and self.desc[1 + self.x, 2 * self.y] == b" ":
            p.append(Actions.LEFT)

        if p:
            return np.random.choice(p)
        else:
            return None

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

    def get_demo(self):
        demo = []
        exp = {}
        exp['s0'] = self.reset()
        while True:
            a = self.go(7,3)
            if a is None:
                break
            else:
                a = np.expand_dims(a, axis=1)
                exp['a'] = a
                exp['s1'] = self.step(a, True)[0]
                exp['o'] = np.expand_dims(1, axis=1)
                demo.append(exp.copy())
                exp['s0'] = exp['s1']
        return demo

    def render(self, goal=(0, 0)):
        outfile = sys.stdout
        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        out[1 + self.x][1 + 2 * self.y] = utils.colorize(out[1 + self.x][1 + 2 * self.y], 'yellow', highlight=True)
        out[1 + goal[0]][1 + 2 * goal[1]] = utils.colorize(out[1 + goal[0]][1 + 2 * goal[1]], 'green', highlight=True)
        outfile.write("\n".join(["".join(row) for row in out])+"\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Up", "Down", "Left","Right",][self.lastaction[0]]))
        else: outfile.write("\n")

if __name__ == '__main__':
    env = Rooms1(args={'--tutoronly': '-1'})
    s = env.reset()
    while True:
        print(env.x, env.y)
        a = env.go(7,5)
        if a is None:
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
