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

        self.num_actions = wrapper.action_dim
        self.initModels()
        self.initTargetModels()

        self.names = ['s0', 'a', 's1', 'r1', 'g', 'o', 't']
        self.buffer = PrioritizedReplayBuffer(limit=int(5e4), names=self.names)
        self.batch_size = int(args['--batchsize'])
        self.train_step = 1

    def initModels(self):

        ### Inputs
        S = Input(shape=self.wrapper.state_dim)
        A = Input(shape=(1,), dtype='uint8')
        G = Input(shape=self.wrapper.goal_dim)
        TARGETS = Input(shape=(1,))
        O = Input(shape=(1,))
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

        loss = K.mean(l2_loss + imit_loss, axis=0)
        inputs = [S, A, G, TARGETS, O]
        updates = Adam(lr=0.001).get_updates(loss, self.model.trainable_weights)
        metrics = [loss, qval, l2_loss, imit_loss]
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

    def get_targets_dqn(self, s, r, t, g):
        qvals = self.qvals([s, g])[0]
        actions = np.argmax(qvals, axis=1)
        a1 = np.expand_dims(np.array(actions), axis=1)
        q = self.targetqval([s, g, a1])[0]
        targets = r + (1 - t) * self.gamma * q.squeeze()
        targets = np.clip(targets, self.wrapper.rNotTerm / (1 - self.wrapper.gamma), self.wrapper.rTerm)
        return np.expand_dims(targets, axis=1)

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

    def train_dqn(self):
        samples = self.buffer.sample(self.batch_size)
        if samples is not None:
            s1 = samples['s1']
            r1 = samples['r1']
            t = samples['t']
            g = samples['g']
            targets = self.get_targets_dqn(s1, r1, t, g)
            s0 = samples['s0']
            a = samples['a']
            o = samples['o']
            inputs = [s0, a, g, targets, o]
            loss, qval, l2_loss, imit_loss = self.train(inputs)
            self.train_step += 1
        if self.train_step % 1000 == 0:
            self.target_train()