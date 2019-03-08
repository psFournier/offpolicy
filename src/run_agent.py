from keras.models import load_model
import numpy as np
from envs.playroomReward import PlayroomReward
from utils.util import softmax

model = load_model('../log/cluster/0803/dqn_PlayroomRewardSparse1-v0/20190308130951_578259/model')

env = PlayroomReward(sparsity=1)

s = env.reset()
# x = np.random.randint(env.nR)
# y = np.random.randint(env.nC)
# g = np.array(env.rescale([x,y]))
i = 0
sum = 0
while i < 200:
    # print('state: ', env.x, env.y)
    input = [np.expand_dims(i, axis=0) for i in [s, np.empty(0)]]
    qvals = model.predict(input)[0].squeeze()
    # print('qvals:',qvals)
    print('probs: ', softmax(qvals, theta=2))
    # action = np.argmax(qvals, axis=0)
    action = np.random.choice(range(qvals.shape[0]), p=softmax(qvals, theta=10))
    # print('action: ', action)
    a = np.expand_dims(action, axis=1)
    s, r, _,_ = env.step(a)
    sum += r
    print('sum: ', sum)
    i += 1
