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

class TB(object):
    def __init__(self, args, wrapper):
        self.wrapper = wrapper

        self.tau = 0.001
        self.a_dim = (1,)
        self.gamma = 0.99
        self.margin = float(args['--margin'])
        self.layers = [int(l) for l in args['--layers'].split(',')]
        self.her = float(args['--her'])
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
        goals = np.expand_dims(trajectory[-1]['goal'], axis=0)
        for i, exp in enumerate(reversed(trajectory)):
            if i==0:
                exp['next'] = None
            else:
                exp['next'] = trajectory[-i]
            if np.random.rand() < self.her:
                goals = np.vstack([goals, exp['s1']])
            exp['goal'] = goals
            exp['terminal'], exp['reward'] = self.wrapper.get_r(exp['s1'], goals)
            self.buffer.append(exp)

    def getNStepSequences(self, exps):

        nStepSeqs = []
        nStepEnds = []

        for exp in exps:

            goalIdx = np.random.randint(exp['goal'].shape[0])
            nStepSeq = [[exp['s0'],
                          exp['a0'],
                          exp['goal'][goalIdx],
                          exp['reward'][goalIdx],
                          exp['terminal'][goalIdx],
                          exp['mu0'],
                          exp['origin']]]

            i = 1
            while i <= self.nstep - 1 and exp['next'] != None and not exp['terminal'][goalIdx]:
                exp = exp['next']
                nStepSeq.append([exp['s0'],
                                  exp['a0'],
                                  exp['goal'][goalIdx],
                                  exp['reward'][goalIdx],
                                  exp['terminal'][goalIdx],
                                  exp['mu0'],
                                  exp['origin']])
                i += 1

            nStepEnds.append([exp['s1'],
                             exp['goal'][goalIdx]])
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

        bootstraps = np.multiply(targetQvalues[-self.batch_size:, :],
                                 actionProbs[-self.batch_size:, :])
        bootstraps = np.sum(bootstraps, axis=1, keepdims=True)

        for i, expe in enumerate(flatExpes):
            expe += [actionProbs[i, :], targetQvalues[i, :]]

        return nStepExpes, bootstraps

    def get_targets(self, nStepExpes, bootstraps):

        targets = []
        states = []
        actions = []
        goals = []
        origins = []
        ros = []
        ro = 1
        for i in range(len(nStepExpes)):
            returnVal = bootstraps[i]
            nStepExpe = nStepExpes[i]
            for j in reversed(range(len(nStepExpe))):
                (s0, a0, g, r, t, mu, o, pis, q) = nStepExpe[j]
                returnVal = r + self.gamma * (1 - t) * returnVal
                if int(self.args['--targetClip']):
                    returnVal = np.clip(returnVal,
                                        self.wrapper.rNotTerm / (1 - self.wrapper.gamma),
                                        self.wrapper.rTerm)
                targets.append(returnVal)
                states.append(s0)
                actions.append(a0)
                goals.append(g)
                origins.append(o)
                q[a0] = returnVal
                returnVal = np.sum(np.multiply(pis, q), keepdims=True)
                ro *= pis[a0] / mu
                ros.append(ro)

        res = [np.array(x) for x in [states, actions, goals, targets, origins, ros]]
        return res

    def train_dqn(self):
        train_stats = {}
        exps = self.buffer.sample(self.batch_size)
        # names = ['s0', 'a0', 's1', 'goal', 'origin', 'term', 'next', 'reached', 'p0', 'weights']
        # if self.alpha != 0:
        #     names.append('indices')
        nStepExpes, nStepEnds = self.getNStepSequences(exps)
        nStepExpes, bootstraps = self.getQvaluesAndBootstraps(nStepExpes, nStepEnds)
        states, actions, goals, targets, origins, ros = self.get_targets(nStepExpes, bootstraps)
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
        if self.train_step < 1e5:
            eps = 1 + ((0.1 - 1) / 1e5) * self.train_step
        else:
            eps = 0.1
        return eps