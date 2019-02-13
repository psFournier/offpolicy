from keras.models import load_model
import numpy as np
from envs.rooms1 import Rooms1
from utils.util import softmax

model = load_model('../log/local/3_Rooms1-v0/20190212150118_358700/log_steps/model')

env = Rooms1(args={'--tutoronly': '-1'})

s = env.reset()
x = np.random.randint(env.nR)
y = np.random.randint(env.nC)
g = np.array(env.rescale([x,y]))
i = 0
print(g, '\n')
while np.linalg.norm(s-g, axis=-1) > 0.001 and i < 500:
    env.render(goal=(x,y))
    input = [np.expand_dims(i, axis=0) for i in [s, g]]
    qvals = model.predict(input)[0].squeeze()
    print(qvals)
    print(softmax(qvals, theta=2))
    # action = np.argmax(qvals, axis=0)
    action = np.random.choice(range(qvals.shape[0]), p=softmax(qvals, theta=1))
    a = np.expand_dims(action, axis=1)
    s = env.step(a)[0]
    i += 1
env.render(goal=(x,y))
