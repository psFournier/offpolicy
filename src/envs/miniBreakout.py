import numpy as np
from gym import Env
from random import randint

################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian (ttian@ualberta.ca)                                                                                #
################################################################################################################


#####################################################################################################################
# Env
#
# The player controls a paddle on the bottom of the screen and must bounce a ball tobreak 3 rows of bricks along the
# top of the screen. A reward of +1 is given for each brick broken by the ball.  When all bricks are cleared another 3
# rows are added. The ball travels only along diagonals, when it hits the paddle it is bounced either to the left or
# right depending on the side of the paddle hit, when it hits a wall or brick it is reflected. Termination occurs when
# the ball hits the bottom of the screen. The balls direction is indicated by a trail channel.
#
#####################################################################################################################
class MiniBreakout(Env):
    def __init__(self, ramping=None, seed=None):
        self.channels = {
            'paddle': 0,
            'ball': 1,
            'trail': 2,
            'brick': 3,
        }
        self.action_map = ['n', 'l', 'u', 'r', 'd', 'f']
        self.random = np.random.RandomState(seed)
        self.reset()

    # Update environment according to agent action
    def step(self, a):
        r = 0
        a = self.action_map[a]

        # Resolve player action
        if (a == 'l'):
            self.pos = max(0, self.pos - 1)
        elif (a == 'r'):
            self.pos = min(9, self.pos + 1)

        # Update ball position
        self.last_x = self.ball_x
        self.last_y = self.ball_y
        if (self.ball_dir == 0):
            new_x = self.ball_x - 1
            new_y = self.ball_y - 1
        elif (self.ball_dir == 1):
            new_x = self.ball_x + 1
            new_y = self.ball_y - 1
        elif (self.ball_dir == 2):
            new_x = self.ball_x + 1
            new_y = self.ball_y + 1
        elif (self.ball_dir == 3):
            new_x = self.ball_x - 1
            new_y = self.ball_y + 1

        strike_toggle = False
        if (new_x < 0 or new_x > 9):
            if (new_x < 0):
                new_x = 0
            if (new_x > 9):
                new_x = 9
            self.ball_dir = [1, 0, 3, 2][self.ball_dir]
        if (new_y < 0):
            new_y = 0
            self.ball_dir = [3, 2, 1, 0][self.ball_dir]
        elif (self.brick_map[new_y, new_x] == 1):
            strike_toggle = True
            if (not self.strike):
                r += 1
                self.strike = True
                self.brick_map[new_y, new_x] = 0
                new_y = self.last_y
                self.ball_dir = [3, 2, 1, 0][self.ball_dir]
        elif (new_y == 9):
            if (np.count_nonzero(self.brick_map) == 0):
                self.brick_map[1:4, :] = 1
            if (self.ball_x == self.pos):
                self.ball_dir = [3, 2, 1, 0][self.ball_dir]
                new_y = self.last_y
            elif (new_x == self.pos):
                self.ball_dir = [2, 3, 0, 1][self.ball_dir]
                new_y = self.last_y
            else:
                self.terminal = True

        if (not strike_toggle):
            self.strike = False

        self.ball_x = new_x
        self.ball_y = new_y
        return self.state(), r, self.terminal, {}

    # Query the current level of the difficulty ramp, difficulty does not ramp in this game, so return None
    def difficulty_ramp(self):
        return None

        # Process the game-state into the 10x10xn state provided to the agent and return

    def state(self):
        state = np.zeros((10, 10, len(self.channels)))
        state[self.ball_y, self.ball_x, self.channels['ball']] = 1
        state[9, self.pos, self.channels['paddle']] = 1
        state[self.last_y, self.last_x, self.channels['trail']] = 1
        state[:, :, self.channels['brick']] = self.brick_map
        return state

    # Reset to start state for new episode
    def reset(self):
        self.ball_y = 3
        ball_start = self.random.choice(2)
        self.ball_x, self.ball_dir = [(0, 2), (9, 3)][ball_start]
        self.pos = 4
        self.brick_map = np.zeros((10, 10))
        self.brick_map[1:4, :] = 1
        self.strike = False
        self.last_x = self.ball_x
        self.last_y = self.ball_y
        self.terminal = False
        return self.state()

    # Dimensionality of the game-state (10x10xn)
    def state_shape(self):
        return [10, 10, len(self.channels)]

# if __name__ == '__main__':
#     env = MiniBreakout()
#     s = env.reset()
#     # while True:
#     #     print(s)
#     #     a, done = env.opt_action(task)
#     #     if done:
#     #         break
#     #     else:
#     #         a = np.expand_dims(a, axis=1)
#     #         s = env.step(a, True)[0]
#     step = 0
#     ep_step = 0
#     dones = 0
#     task = 0
#     prop = 0.99
#     while step < 50000:
#         done = (env.objects[task].s == env.objects[task].high)
#         if env.objects[1].s == env.objects[1].high:
#             dones += 1
#         if done or ep_step >= 200:
#             s = env.reset()
#             ep_step = 0
#             task = np.random.choice([0, 1], p=[prop, 1 - prop])
#         else:
#             a, _ = env.opt_action(task)
#             a = np.expand_dims(a, axis=1)
#             s = env.step(a)[0]
#             ep_step += 1
#             step += 1
#     print(dones)
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
