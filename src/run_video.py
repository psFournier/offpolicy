from keras.models import load_model
import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np
from env_wrappers.registration import registry, load
from env_wrappers import Base

log_dir = '../../log/local/ddpg_Reacher1D-v0/20181012120559_241300/log_steps/'
model = load_model(os.path.join(log_dir, 'actor_model.h5'))
spec = registry.spec('Reacher1D-v0')
env = spec.make()
if env.spec.wrapper_entry_point is not None:
    wrapper_cls = load(env.spec.wrapper_entry_point)
    env = wrapper_cls(env, {'--gamma': 0.99})
else:
    env = Base(env, {'--gamma': 0.99})

vid_dir = os.path.join(log_dir, 'videos')
os.makedirs(vid_dir, exist_ok=True)
base_path = os.path.join(vid_dir, 'video_init')
rec = VideoRecorder(env, base_path=base_path)

state = env.reset()
rec.capture_frame()
reward_sum = 0
for k in range(500):
    env.render(mode='human')
    action = model.predict(np.reshape(state, (1, env.state_dim[0])))
    action = np.clip(action, env.action_space.low, env.action_space.high)
    state = env.env.step(action[0])[0]
    rec.capture_frame()

