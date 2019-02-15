import numpy as np
from utils.segment_tree import SumSegmentTree, MinSegmentTree

class ReplayBuffer(object):
    def __init__(self, limit):
        self._storage = []
        self._next_idx = 0
        self._limit = limit

    def __len__(self):
        return len(self._storage)

    def append(self, item):
        if self._next_idx >= len(self._storage):
            self._storage.append(item)
        else:
            self._storage[self._next_idx] = item
        self._next_idx = (self._next_idx + 1) % self._limit

    def sample(self, batchsize, idxs=None, beta=None):
        if idxs is None:
            idxs = [np.random.randint(0, len(self._storage) - 1) for _ in range(batchsize)]
        exps = []
        for i in idxs:
            exps.append(self._storage[i])
            exps[-1]['weights'] = np.expand_dims(1, axis=1)
        return exps

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, limit, alpha=0):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(limit=limit)
        self.alpha = alpha

        self.epsilon = 1e-6


        it_capacity = 1
        while it_capacity < limit:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self.max_priority = 1.

    def append(self, buffer_item):
        idx = self._next_idx
        super().append(buffer_item)
        self._it_sum[idx] = self.max_priority ** self.alpha
        self._it_min[idx] = self.max_priority ** self.alpha

    def _sample_proportional(self, batch_size):
        res = []
        sum = self._it_sum.sum(0, len(self._storage) - 1)
        for _ in range(batch_size):
            mass = np.random.random() * sum
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, idxs=None, beta=0.4):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        """
        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(np.expand_dims(weight / max_weight, axis=1))

        exps = super(PrioritizedReplayBuffer, self).sample(batch_size, idxes)
        for exp, idx, weight in zip(exps, idxes, weights):
            exp['indices'] = np.expand_dims(idx, axis=1)
            exp['weights'] = np.expand_dims(weight, axis=1)

        return exps


    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        idxes = list(idxes.squeeze())
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            priority = priority + self.epsilon
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self.alpha
            self._it_min[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)
