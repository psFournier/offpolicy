from keras.models import Model
from keras.initializers import RandomUniform, lecun_uniform
from keras.regularizers import l2
from keras.layers import Dense, Input, Lambda, Reshape, Dropout
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.merge import concatenate, multiply, add, subtract, maximum, Dot
import numpy as np
from prioritizedReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer

from keras.losses import mse
import tensorflow as tf
from utils.util import softmax
import time

class Dqn(object):
    def __init__(self, args, wrapper):
        self.wrapper = wrapper

        self.tau = 0.001
        self.a_dim = (1,)
        self.gamma = 0.99
        self.margin = float(args['--margin'])
        self.layers = [int(l) for l in args['--layers'].split(',')]
        self.her_p = int(args['--her_p'])
        self.nstep = int(args['--nstep'])

        self.num_actions = wrapper.action_dim
        self.initModels()
        self.initTargetModels()

        self.names = ['s0', 'a0',  's1', 'goal', 'origin', 'term', 'next', 'reached']
        self.alpha = float(args['--alpha'])
        if self.alpha == 0:
            self.buffer = ReplayBuffer(limit=int(5e4), names=self.names)
        else:
            self.buffer = PrioritizedReplayBuffer(limit=int(5e4), names=self.names, alpha=self.alpha)
        self.batch_size = 64
        self.train_step = 1

    def initModels(self):

        ### Inputs
        S = Input(shape=self.wrapper.state_dim)
        A = Input(shape=(1,), dtype='uint8')
        G = Input(shape=self.wrapper.goal_dim)
        TARGETS = Input(shape=(1,))
        O = Input(shape=(1,))
        W = Input(shape=(1,))
        # MCR = Input(shape=(1,), dtype='float32')

        ### Q values and action models
        qvals = self.create_critic_network(S, G)
        self.model = Model([S, G], qvals)
        self.qvals = K.function(inputs=[S, G], outputs=[qvals], updates=None)
        actionProbs = K.softmax(qvals)
        self.actionProbs = K.function(inputs=[S, G], outputs=[actionProbs], updates=None)
        actionFilter = K.squeeze(K.one_hot(A, self.num_actions), axis=1)
        qval = K.sum(actionFilter * qvals, axis=1, keepdims=True)
        self.qval = K.function(inputs=[S, G, A], outputs=[qval], updates=None)

        ### DQN loss
        td_errors = qval - TARGETS
        l2_loss = K.square(td_errors)

        ### Large margin loss
        # qvalWidth = K.max(qvals, axis=1, keepdims=True) - K.min(qvals, axis=1, keepdims=True)
        onehot = 1 - K.squeeze(K.one_hot(A, self.num_actions), axis=1)
        imit_loss = (K.max(qvals + self.margin * onehot, axis=1, keepdims=True) - qval) * O

        loss = K.dot(K.transpose(W), l2_loss + imit_loss) / K.sum(W, axis=0)
        inputs = [S, A, G, TARGETS, O, W]
        updates = Adam(lr=0.001).get_updates(loss, self.model.trainable_weights)
        metrics = [loss, qval, td_errors, imit_loss]
        self.train = K.function(inputs, metrics, updates)

    def initTargetModels(self):
        S = Input(shape=self.wrapper.state_dim)
        G = Input(shape=self.wrapper.goal_dim)
        A = Input(shape=(1,), dtype='uint8')
        Tqvals = self.create_critic_network(S, G)
        self.targetmodel = Model([S, G], Tqvals)

        actionFilter = K.squeeze(K.one_hot(A, self.num_actions), axis=1)
        Tqval = K.sum(actionFilter * Tqvals, axis=1, keepdims=True)
        self.targetqval = K.function(inputs=[S, G, A], outputs=[Tqval], updates=None)

        self.target_train()

    def target_train(self):
        self.targetmodel.set_weights(self.model.get_weights())

    def create_critic_network(self, S, G):
        h = concatenate([subtract([S, G]), S])
        for l in self.layers:
            h = Dense(l, activation="relu",
                      kernel_initializer=lecun_uniform(),
                      kernel_regularizer=l2(0.01))(h)
        Q_values = Dense(self.num_actions,
                         activation='linear',
                         kernel_initializer=RandomUniform(minval=-3e-4, maxval=3e-4),
                         kernel_regularizer=l2(0.01),
                         bias_initializer=RandomUniform(minval=-3e-4, maxval=3e-4))(h)
        return Q_values

    def process_trajectory(self, trajectory):
        l = len(trajectory)
        idxs = np.random.choice(l, size=min(self.her_p, l), replace=False)
        for i, exp in enumerate(trajectory):
            if i==len(trajectory)-1:
                exp['next'] = None
            else:
                exp['next'] = trajectory[i+1]
            exp['reached'] = [trajectory[idx]['s1'] for idx in idxs if idx >= i]
            self.buffer.append(exp)

            ### Hindsight exp replay
            # for vg in virtual_g:
            #     exp['g'] = vg
            #     exp['t'], exp['r1'] = self.wrapper.get_r(exp['s1'], vg)
            #     res.append(exp.copy())
            # if np.random.rand() < self.her_p:
            #     virtual_g.append(exp['s0'])

    # def get_targets_dqn(self, s, r, t, g):
    #     qvals = self.qvals([s, g])[0]
    #     actions = np.argmax(qvals, axis=1)
    #     a1 = np.expand_dims(np.array(actions), axis=1)
    #     q = self.targetqval([s, g, a1])[0]
    #     targets = r + (1 - t) * self.gamma * q.squeeze()
    #     targets = np.clip(targets, self.wrapper.rNotTerm / (1 - self.wrapper.gamma), self.wrapper.rTerm)
    #     return np.expand_dims(targets, axis=1)

    # def get_targets_dqn(self, samples):
    #     s = samples['s1']
    #     goal = samples['goal']
    #
    #     s1, g, G, t, gamma = [], [], [], [], []
    #     for sample in samples:
    #         goals = [sample['goal']]+sample['reached']
    #         goal = goals[np.random.choice(len(goals))]
    #         term, target = self.wrapper.get_r(sample['s1'], goal)
    #         i=1
    #         while sample['next'] is not None and i<self.nstep:
    #             sample = sample['next']
    #             term, reward = self.wrapper.get_r(sample['s1'], goal)
    #             target += (self.gamma ** i) * reward
    #             i += 1
    #         s1.append(sample['s1'])
    #         G.append(target)
    #         t.append(term)
    #         g.append(goal)
    #         gamma.append(self.gamma ** i)
    #     a_s1, a_g = np.array(s1), np.array(g)
    #     qvals = self.qvals([a_s1, a_g])[0]
    #     actions = np.argmax(qvals, axis=1)
    #     action = np.expand_dims(np.array(actions), axis=1)
    #     bootstrap = self.targetqval([a_s1, a_g, action])[0]
    #     a_G = np.array(G) + (1 - np.array(t)) * np.array(gamma) * bootstrap.squeeze()
    #     a_G = np.expand_dims(a_G, axis=1)
    #     return a_g, a_G

    def get_targets(self, samples):

        g = []
        for i, goal in enumerate(samples['goal']):
            reached = samples['reached'][i]
            if list(reached) and np.random.rand() < 0.5:
                g.append(reached[np.random.choice(len(reached))])
            else:
                g.append(goal)
        g = np.array(g)

        s = samples['s1']
        t, r = self.wrapper.get_r(s, g)
        G = r.copy()
        gamma = np.ones_like(r)

        next = samples['next']
        for i in range(self.nstep - 1):
            indices = np.where(next != None)
            for idx in indices[0]:
                s[idx] = next[idx]['s1']
                next[idx] = next[idx]['next']
            t[indices], r[indices] = self.wrapper.get_r(s[indices], g[indices])
            gamma[indices] *= self.gamma
            G[indices] += gamma[indices] * r[indices]

        qvals = self.qvals([s, g])[0]
        actions = np.argmax(qvals, axis=1)
        an = np.expand_dims(np.array(actions), axis=1)
        bootstrap = self.targetqval([s, g, an])[0]
        G += (1 - t) * self.gamma * gamma * bootstrap.squeeze()
        G = np.clip(G, self.wrapper.rNotTerm / (1 - self.wrapper.gamma), self.wrapper.rTerm)
        return np.expand_dims(G, axis=1), g

    # def get_targets_dqn(self, samples):
    #     s = samples['s1']
    #     goal = samples['goal']
    #
    #     t, r = self.wrapper.get_r(s, goal)
    #     G = r.copy()
    #     gamma = np.ones_like(r)
    #
    #     ### n-step boostrap
    #     next = samples['next']
    #     for i in range(self.nstep - 1):
    #         tn = t.copy()
    #         for j in range(self.batch_size):
    #             if next[j] is not None:
    #                 s[j] = next[j]['s1']
    #                 tn[j], r[j] = self.wrapper.get_r(s[j], goal[j])
    #                 next[j] = next[j]['next']
    #                 gamma[j] *= self.gamma
    #             else:
    #                 s[j], tn[j], r[j] = s[j], t[j], 0
    #         G += (1 - t) * r * gamma
    #         t = tn
    #
    #     qvals = self.qvals([s, goal])[0]
    #     actions = np.argmax(qvals, axis=1)
    #     an = np.expand_dims(np.array(actions), axis=1)
    #     bootstrap = self.targetqval([s, goal, an])[0]
    #     G += (1 - t) * self.gamma * gamma * bootstrap.squeeze()
    #     return np.expand_dims(G, axis=1)

    # def train_dqn(self):
    #     samples = self.buffer.sample(self.batch_size)
    #     if samples is not None:
    #         # goal, targets = self.get_inputs_dqn(samples)
    #         targets = self.get_targets_dqn(samples)
    #         goal = np.array([s['goal'] for s in samples])
    #         s0 = np.array([s['s0'] for s in samples])
    #         a0 = np.array([s['a0'] for s in samples])
    #         origin = np.array([s['origin'] for s in samples])
    #         inputs = [s0, a0, goal, targets, origin]
    #         loss, qval, l2_loss, imit_loss = self.train(inputs)
    #         self.train_step += 1
    #     if self.train_step % 1000 == 0:
    #         self.target_train()

    def train_dqn(self):
        beta = min(0.4 + (1 - 0.4) * self.train_step / 5e5, 1)
        samples = self.buffer.sample(self.batch_size, beta=beta)
        if samples is not None:
            targets, goal = self.get_targets(samples)
            s0 = samples['s0']
            a0 = samples['a0']
            origin = samples['origin']
            weights = samples['weights']
            inputs = [s0, a0, goal, targets, origin, weights]
            loss, qval, td_errors, imit_loss = self.train(inputs)
            if self.alpha != 0:
                self.buffer.update_priorities(samples['indices'], np.abs(td_errors.squeeze()))
            self.train_step += 1
        if self.train_step % 1000 == 0:
            self.target_train()