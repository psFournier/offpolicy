import tensorflow as tf
import numpy as np
from utils.util import build_logger
from env_wrappers.registration import make
from docopt import docopt
from utils.util import softmax
from dqn import Dqn
import time
import os
from keras.models import load_model

help = """

Usage: 
  main.py --env=<ENV> --agent=<AGENT> [options]

Options:
  --seed SEED              Random seed
  --inv_grad YES_NO        Gradient inversion near action limits [default: 1]
  --max_steps VAL          Maximum total steps [default: 800000]
  --ep_steps VAL           Maximum episode steps [default: 200]
  --ep_tasks VAL           Maximum episode tasks [default: 1]
  --log_dir DIR            Logging directory [default: /home/pierre/PycharmProjects/offpolicy/log/local/]
  --eval_freq VAL          Logging frequency [default: 5000]
  --margin VAL             Large margin loss margin [default: 1]
  --gamma VAL              Discount factor [default: 0.99]
  --batchsize VAL          Batch size [default: 64]
  --wimit VAL              Weight for imitaiton loss with imitaiton [default: 1]
  --rnd_demo VAL           Amount of stochasticity in the tutor's actions [default: 0]
  --network VAL            network type [default: 0]
  --filter VAL             network type [default: 0]
  --prop_demo VAL             network type [default: 0.02]
  --freq_demo VAL             network type [default: 100000000]
  --lrimit VAL             network type [default: 0.001]
  --rndv VAL               [default: 0]
  --demo VAL               [default: -1]
  --tutoronly VAL          [default: -1]
  --initq VAL              [default: -100]
  --layers VAL             [default: 400,300]
  --her_p VAL              [default: 0.01]
  --nstep VAL              [default: 1]
  --alpha VAL              [default: 0]
"""

if __name__ == '__main__':

    args = docopt(help)

    logger = build_logger(args)
    env, wrapper = make(args['--env'], args)
    env_test, wrapper_test = make(args['--env'], args)

    seed = args['--seed']
    if seed is not None:
        seed = int(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        env.seed(seed)
        env_test.seed(seed)

    agent = Dqn(args, wrapper)

    model = load_model('../log/local/3_Rooms1-v0/20190212112608_490218/log_steps/model')
    demo = [int(f) for f in args['--demo'].split(',')]
    imit_steps = int(float(args['--freq_demo']) * float(args['--prop_demo']))
    max_episode_steps = int(args['--ep_steps'])

    env_step = 1
    episode_step = 0
    trajectory = []
    state = env.reset()
    goal = wrapper.get_g()
    t0 = time.time()
    while env_step < int(args['--max_steps']):

        # if env_step % int(args['--freq_demo']) == 0:
        #     for _ in range(25):
        #         s = env_test.reset()
        #         x = np.random.randint(env_test.nR)
        #         y = np.random.randint(env_test.nC)
        #         g = np.array(env_test.rescale([x, y]))
        #         i = 0
        #         demo = []
        #         while np.linalg.norm(s - g, axis=-1) > 0.001 and i < 200:
        #             exp = {'s0': s.copy()}
        #             input = [np.expand_dims(i, axis=0) for i in [s, g]]
        #             qvals = model.predict(input)[0].squeeze()
        #             action = np.argmax(qvals, axis=0)
        #             a = np.expand_dims(action, axis=1)
        #             s = env_test.step(a)[0]
        #             i += 1
        #             exp['a0'], exp['s1'], exp['origin'] = a, s.copy(), np.expand_dims(1, axis=1)
        #             demo.append(exp.copy())
        #         agent.process_trajectory(demo)

        exp = {'s0': state.copy(), 'goal': goal.copy()}

        input = [np.expand_dims(i, axis=0) for i in [state, goal]]
        qvals = agent.qvals(input)[0].squeeze()
        probs = softmax(qvals, theta=min(1/1e4*env_step,1))
        action = np.random.choice(range(qvals.shape[0]), p=probs)
        a = np.expand_dims(action, axis=1)
        state = env.step(a)[0]
        term, _ = wrapper.get_r(state, goal)

        exp['a0'], exp['term'], exp['s1'], exp['origin'] = a, term, state.copy(), np.expand_dims(0, axis=1)
        # exp['p0'] = probs[action]

        trajectory.append(exp.copy())
        agent.train_dqn()

        env_step += 1
        episode_step += 1

        if term or episode_step >= max_episode_steps:
            agent.process_trajectory(trajectory)
            trajectory.clear()
            state = env.reset()
            goal = wrapper.get_g()
            episode_step = 0

        if env_step % int(args['--eval_freq'])== 0:
            logger.logkv('step', env_step)
            R = 0
            n=10
            for i in range(n):
                term_eval, ep_step_eval = 0, 0
                state_eval = env_test.reset()
                x = np.random.randint(env_test.nR)
                y = np.random.randint(env_test.nC)
                goal_eval = np.array(env_test.rescale([x, y]))
                while not term_eval and ep_step_eval < max_episode_steps:
                    input = [np.expand_dims(i, axis=0) for i in [state_eval, goal_eval]]
                    qvals = agent.qvals(input)[0].squeeze()
                    action = np.argmax(qvals)
                    a = np.expand_dims(action, axis=1)
                    state_eval = env_test.step(a)[0]
                    term_eval, r_eval = wrapper_test.get_r(state_eval, goal_eval)
                    ep_step_eval += 1
                    R += r_eval
            logger.logkv('average return', R / n)
            logger.dumpkvs()
            t1 = time.time()
            print(t1- t0)
            t0 = t1
            agent.model.save(os.path.join(logger.get_dir(), 'model'), overwrite=True)







