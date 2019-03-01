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
from utils.util import softmax, egreedy
import time

class Dqn3(object):
    def __init__(self, args, wrapper):
        self.wrapper = wrapper

        self.tau = 0.001
        self.a_dim = (1,)
        self._gamma = 0.99
        self._lambda = float(args['--lambda'])
        self.margin = float(args['--margin'])
        self.layers = [int(l) for l in args['--layers'].split(',')]
        self.nstep = int(args['--nstep'])
        self.args = args

        self.num_actions = wrapper.action_dim
        self.initModels()
        self.initTargetModels()

        self.alpha = float(args['--alpha'])
        if self.alpha == 0:
            self.buffer = ReplayBuffer(limit=int(5e4))
        else:
            self.buffer = PrioritizedReplayBuffer(limit=int(5e4), alpha=self.alpha)
        self.batch_size = int(64 / self.nstep)
        self.train_step = 0

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
        targetQvals = self.create_critic_network(S, G)
        self.targetmodel = Model([S, G], targetQvals)
        self.targetqvals = K.function(inputs=[S, G], outputs=[targetQvals], updates=None)

        actionFilter = K.squeeze(K.one_hot(A, self.num_actions), axis=1)
        targetQval = K.sum(actionFilter * targetQvals, axis=1, keepdims=True)
        self.targetqval = K.function(inputs=[S, G, A], outputs=[targetQval], updates=None)

        self.target_train()

    def target_train(self):
        self.targetmodel.set_weights(self.model.get_weights())

    def create_critic_network(self, S, G):
        h = concatenate([S, G])
        for l in self.layers:
            h = Dense(l, activation="relu",
                      kernel_initializer=lecun_uniform(),
                      kernel_regularizer=l2(0.01))(h)
        ValAndAdv = Dense(self.num_actions + 1,
                         activation='linear',
                         kernel_initializer=RandomUniform(minval=-3e-4, maxval=3e-4),
                         kernel_regularizer=l2(0.01),
                         bias_initializer=RandomUniform(minval=-3e-4, maxval=3e-4))(h)
        Q_values = Lambda(lambda a: K.expand_dims(a[:, 0], axis=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True, axis=1),
                   output_shape=(self.num_actions,))(ValAndAdv)
        return Q_values

    def act(self, exp):

        input = self.wrapper.make_input(exp)
        qvals = self.qvals(input)[0].squeeze()

        if self.args['--exp'] == 'softmax':
            probs = softmax(qvals, theta=self.theta)
        elif self.args['--exp'] == 'egreedy':
            probs = egreedy(qvals, eps=self.eps)
        else:
            raise RuntimeError

        action = np.random.choice(range(qvals.shape[0]), p=probs)
        # action = np.argmax(qvals)

        return np.expand_dims(action, axis=1), probs

    def process_trajectory(self, trajectory):
        trajectory = self.wrapper.process_trajectory(trajectory)
        for exp in trajectory:
            self.buffer.append(exp)

    def getNStepSequences(self, exps):

        nStepSeqs = []
        nStepEnds = []

        # NE MARCHE QUE POUR PLAYROOM POUR L'INSTANT
        idx = np.random.randint(2, self.wrapper.state_dim[0])
        v = self.wrapper.vs[idx - 2]
        g = np.zeros(self.wrapper.state_dim)
        l, h = self.wrapper.unwrapped.low[idx], self.wrapper.unwrapped.high[idx]
        g[idx] = (np.random.randint(l + 1, h) - l) / (h - l)
        goal = np.hstack([g, v])
        for exp in exps:
            exp['goal'] = goal
            exp = self.wrapper.get_r(exp)
            nStepSeq = [[exp['s0'],
                          exp['a0'],
                          exp['goal'],
                          exp['reward'],
                          exp['terminal'],
                          None,
                          exp['origin']]]

            i = 1
            while i <= self.nstep - 1 and exp['next'] != None and not exp['terminal']:
                exp = exp['next']
                exp['goal'] = goal
                exp = self.wrapper.get_r(exp)
                nStepSeq.append([exp['s0'],
                             exp['a0'],
                             exp['goal'],
                             exp['reward'],
                             exp['terminal'],
                             None,
                             exp['origin']])
                i += 1

            nStepEnds.append([exp['s1'],
                              exp['goal']])
            nStepSeqs.append(nStepSeq)

        return nStepSeqs, nStepEnds

    def getQvaluesAndBootstraps(self, nStepExpes, nStepEnds):

        flatExpes = [expe for nStepExpe in nStepExpes for expe in nStepExpe]
        states = np.array([expe[0] for expe in flatExpes])
        goals = np.array([expe[2] for expe in flatExpes])

        endStates = np.array([end[0] for end in nStepEnds])
        endGoals = np.array([end[1] for end in nStepEnds])
        input = [np.vstack([states, endStates]), np.vstack([goals, endGoals])]
        qValues = self.qvals(input)[0]
        targetQvalues = self.targetqvals(input)[0]

        if self.args['--exp'] == 'softmax':
            actionProbs = softmax(qValues, axis=1, theta=self.theta)
        elif self.args['--exp'] == 'egreedy':
            actionProbs = egreedy(qValues, eps=self.eps)
        else:
            raise RuntimeError

        for i, expe in enumerate(flatExpes):
            expe += [actionProbs[i, :], targetQvalues[i, :]]

        for i in range(len(nStepExpes)):
            nStepExpes[i].append([actionProbs[-self.batch_size + i, :], targetQvalues[-self.batch_size + i, :]])

        return nStepExpes

    def getTargetsSumTD(self, nStepExpes):
        targets = []
        states = []
        actions = []
        goals = []
        origins = []
        ros = []
        for nStepExpe in nStepExpes:
            tdErrors = []
            cs = []
            qs = []

            for expe1, expe2 in zip(nStepExpe[:-1], nStepExpe[1:]):

                s0, a0, g, r, t, mu, o, pis, qt = expe1
                states.append(s0)
                actions.append(a0)
                goals.append(g)
                origins.append(o)
                q = qt[a0]
                qs.append(q)
                pisNext, qtNext = expe2[-2:]

                ### Calcul des one-step td errors variable selon la méthode
                if self.args['--bootstrap'] == 'expectation':
                    b = np.sum(np.multiply(qtNext, pisNext), keepdims=True)
                else:
                    amax = np.argwhere(pisNext == np.max(pisNext)).flatten().tolist()
                    b = qtNext[np.random.choice(amax)]
                b = r + (1 - t) * self._gamma * b

                if int(self.args['--targetClip']):
                    b = np.clip(b, self.wrapper.rNotTerm / (1 - self._gamma), self.wrapper.rTerm)

                tdErrors.append((b - q).squeeze())

                ### Calcul des ratios variable selon la méthode
                if self.args['--IS'] == 'no':
                    cs.append(self._gamma * self._lambda)
                elif self.args['--IS'] == 'standard':
                    ro = pis[a0] / mu
                    cs.append(ro * self._gamma * self._lambda)
                elif self.args['--IS'] == 'retrace':
                    ro = pis[a0] / mu
                    cs.append(min(1, ro) * self._gamma * self._lambda)
                elif self.args['--IS'] == 'tb':
                    cs.append(pis[a0] * self._gamma * self._lambda)
                else:
                    raise RuntimeError

            deltas = []
            for i in range(len(tdErrors) - 1):
                deltas.append(np.sum(np.multiply(tdErrors[i+1:], np.cumprod(cs[i+1:]))))
            deltas.append(0)
            targets += [q + delta + tdError for q, delta, tdError in zip(qs, deltas, tdErrors)]
            ros += cs

        res = [np.array(x) for x in [states, actions, goals, targets, origins, ros]]
        return res

    def train_dqn(self):
        train_stats = {}
        exps = self.buffer.sample(self.batch_size)
        # names = ['s0', 'a0', 's1', 'goal', 'origin', 'term', 'next', 'reached', 'p0', 'weights']
        # if self.alpha != 0:
        #     names.append('indices')
        nStepExpes, nStepEnds = self.getNStepSequences(exps)
        nStepExpes = self.getQvaluesAndBootstraps(nStepExpes, nStepEnds)
        states, actions, goals, targets, origins, ros = self.getTargetsSumTD(nStepExpes)
        train_stats['target_mean'] = np.mean(targets)
        train_stats['ro'] = np.mean(ros)
        inputs = [states, actions, goals, targets, origins, np.ones_like(targets)]
        loss, qval, td_errors, imit_loss = self.train(inputs)
        self.train_step += 1
        if self.train_step % 1000 == 0:
            self.target_train()
        return train_stats

    @property
    def theta(self):
        return 1

    @property
    def eps(self):
        if self.train_step < 1e4:
            eps = 1 + ((0.1 - 1) / 1e4) * self.train_step
        else:
            eps = 0.1
        return eps