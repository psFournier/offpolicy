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
    def high(self):
        return len(self.dep)

    @property
    def low(self):
        return 0

class PlayroomBig(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, multistart=True, size=20, N=4, L=5):
        self.size = size
        self.multistart = multistart
        self.N = N
        self.L = L
        self.objects = []
        for i in range(self.N):
            x, y = np.random.randint(1, self.size-1, size=2)
            l = [(x+i, y+j) for i in [-1, 0, 1] for j in [-1, 0, 1]]
            np.random.shuffle(l)
            self.objects.append(Obj(self,
                                    dep=l[:L]))
        self.initialize()

    def initialize(self):
        if self.multistart:
            self.x = np.random.randint(self.size)
            self.y = np.random.randint(self.size)
        else:
            self.x = 0
            self.y = 0
        for obj in self.objects:
            obj.s = 0
        self.lastaction = None

    def step(self, a):
        env_a = a
        if self.lastaction is not None and np.random.rand() < 0.2:
            env_a = self.lastaction
        self.lastaction = a

        if env_a == Actions.UP:
            self.x = min(self.x + 1, self.size - 1)

        if env_a == Actions.DOWN:
            self.x = max(self.x - 1, 0)

        if env_a == Actions.RIGHT:
            self.y = min(self.y + 1, self.size - 1)

        if env_a == Actions.LEFT:
            self.y = max(self.y - 1, 0)

        elif env_a == Actions.TOUCH:
            for obj in self.objects:
                if obj.s < obj.high and (self.x, self.y) == obj.dep[obj.s]:
                    obj.s += 1

        return np.array(self.state), 0, 0, {}

    def reset(self):
        self.initialize()
        return np.array(self.state)

    @property
    def high(self):
        res = [self.size - 1, self.size - 1] + [obj.high for obj in self.objects]
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

    def touch(self, x, y):
        a = self.go(x, y)
        if a is None:
            return Actions.TOUCH, False
        else:
            return a, False

    def opt_action(self, t):
        obj = self.objects[t]
        if obj.s == obj.high:
            return -1, True
        else:
            dep = obj.dep[obj.s]
            return self.touch(dep[0], dep[1])

if __name__ == '__main__':
    env = Playroom(multistart=True)
    s = env.reset()
    # while True:
    #     print(s)
    #     a, done = env.opt_action(task)
    #     if done:
    #         break
    #     else:
    #         a = np.expand_dims(a, axis=1)
    #         s = env.step(a, True)[0]
    step = 0
    ep_step = 0
    dones = 0
    task = 0
    prop = 0.99
    while step < 50000:
        done = (env.objects[task].s == env.objects[task].high)
        if env.objects[1].s == env.objects[1].high:
            dones += 1
        if done or ep_step >= 200:
            s = env.reset()
            ep_step = 0
            task = np.random.choice([0, 1], p=[prop, 1 - prop])
        else:
            a, _ = env.opt_action(task)
            a = np.expand_dims(a, axis=1)
            s = env.step(a)[0]
            ep_step += 1
            step += 1
    print(dones)
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
