import tensorflow as tf
import numpy as np
from utils.util import build_logger
from env_wrappers.registration import make
from docopt import docopt
from utils.util import softmax, egreedy
from dqn import Dqn
from dqn2 import Dqn2
from dqn3 import Dqn3
import time
import os
from keras.models import load_model
from utils.logger import Logger

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
  --eval_freq VAL          Logging frequency [default: 2000]
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
  --initq VAL              [default: 0]
  --layers VAL             [default: 64,64]
  --her VAL                [default: 0]
  --nstep VAL              [default: 1]
  --alpha VAL              [default: 0]
  --IS VAL                 [default: no]
  --exp VAL                [default: softmax]
  --targetClip VAL         [default: 0]
  --lambda VAL             [default: 0]
  --bootstrap VAL          [default: expectation]
  --goal_replay VAL        [default: 0]
"""

if __name__ == '__main__':

    args = docopt(help)

    log_dir = build_logger(args)
    loggerTB = Logger(dir=log_dir,
                      format_strs=['tensorboard_{}'.format(int(args['--eval_freq'])),
                                   'stdout'])
    loggerJSON = Logger(dir=log_dir,
                        format_strs=['json'])
    env, wrapper = make(args['--env'], args)
    env_test, wrapper_test = make(args['--env'], args)

    seed = args['--seed']
    if seed is not None:
        seed = int(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        env.seed(seed)
        env_test.seed(seed)

    agent = Dqn3(args, wrapper)


    stats = {'target_mean': 0,
             'train_step': 0,
             'ro': 0,
             'reward_train': 0,
             'reward_test': 0}

    nb_ep_train = 0
    nb_ep_test = 0
    nb_ep = 0
    t0 = time.time()

    # model = load_model('../log/local/3_Rooms1-v0/20190212112608_490218/log_steps/model')
    demo = [int(f) for f in args['--demo'].split(',')]
    imit_steps = int(float(args['--freq_demo']) * float(args['--prop_demo']))
    max_episode_steps = int(args['--ep_steps'])

    # Put demo data in buffer
    # state = env_test.reset()
    # task = 0
    # demo_step = 0
    # demo_ep_step = 0
    # exp = {}
    # traj = []
    # prop = float(args['--rnd_demo'])
    # while demo_step < 500:
    #     done = (env_test.objects[task].s == env_test.objects[task].high)
    #     if done or demo_ep_step >= 200:
    #         state = env_test.reset()
    #         task = np.random.choice([0, 1], p=[prop, 1 - prop])
    #         for i, exp in enumerate(reversed(traj)):
    #             if i == 0:
    #                 exp['next'] = None
    #             else:
    #                 exp['next'] = traj[-i]
    #             agent.buffer.append(exp)
    #         demo_ep_step = 0
    #     else:
    #         exp['s0'] = state.copy()
    #         a, _ = env_test.opt_action(task)
    #         a = np.expand_dims(a, axis=1)
    #         exp['a0'], exp['mu0'], exp['origin'] = a, None, np.expand_dims(1, axis=1)
    #         state = env_test.step(a)[0]
    #         exp['s1'] = state.copy()
    #         traj.append(exp.copy())
    #         demo_ep_step += 1
    #         demo_step += 1

    env_step = 1
    episode_step = 0
    reward_train = 0
    reward_test = 0
    trajectory = []
    state = env.reset()
    exp = wrapper.reset(state)

    while env_step < int(args['--max_steps']):

        a, probs = agent.act(exp)
        exp['a0'], exp['mu0'], exp['origin'] = a, probs[a], np.expand_dims(0, axis=1)
        state, r, term, info = env.step(a.squeeze())
        exp['s1'] = state.copy()
        exp['terminal'], exp['reward'] = wrapper.get_r(exp['s1'], exp['goal'], r, term)
        env_step += 1
        episode_step += 1

        reward_train += exp['reward']

        trajectory.append(exp.copy())
        exp['s0'] = state

        if len(agent.buffer) > 10000:
            train_stats = agent.train_dqn()
            stats['target_mean'] += train_stats['target_mean']
            stats['train_step'] += 1
            stats['ro'] += train_stats['ro']

        if exp['terminal'] or episode_step >= max_episode_steps:
            nb_ep_train += 1
            stats['reward_train'] += reward_train
            nb_ep += 1
            agent.process_trajectory(trajectory)
            trajectory.clear()
            state = env.reset()
            exp = wrapper.reset(state)
            episode_step = 0
            reward_train = 0
            reward_test = 0


        if env_step % int(args['--eval_freq'])== 0:
            # R = 0
            # n=10
            # for i in range(n):
            #     term_eval, ep_step_eval = 0, 0
            #     state_eval = env_test.reset()
            #     x = np.random.randint(env_test.nR)
            #     y = np.random.randint(env_test.nC)
            #     goal_eval = np.array(env_test.rescale([x, y]))
            #     while not term_eval and ep_step_eval < max_episode_steps:
            #         input = [np.expand_dims(i, axis=0) for i in [state_eval, goal_eval]]
            #         qvals = agent.qvals(input)[0].squeeze()
            #         action = np.argmax(qvals)
            #         a = np.expand_dims(action, axis=1)
            #         state_eval = env_test.step(a)[0]
            #         term_eval, r_eval = wrapper_test.get_r(state_eval, goal_eval)
            #         ep_step_eval += 1
            #         R += r_eval

            # loggerJSON.logkv('goal_freq', stats['goal_freq'])
            for logger in [loggerJSON, loggerTB]:
                logger.logkv('step', env_step)
                logger.logkv('target_mean', stats['target_mean'] / (stats['train_step'] + 1e-5))
                logger.logkv('ro', stats['ro'] / (stats['train_step'] + 1e-5))
                logger.logkv('reward_train', stats['reward_train'] / (nb_ep_train + 1e-5))
                logger.logkv('reward_test', stats['reward_test'] / (nb_ep_test + 1e-5))
                logger.dumpkvs()

            stats['target_mean'] = 0
            stats['ro'] = 0
            stats['train_step'] = 0
            stats['reward_train'] = 0
            nb_ep_train = 0
            stats['reward_test'] = 0
            nb_ep_test = 0

            t1 = time.time()
            print(t1- t0)
            t0 = t1
            agent.model.save(os.path.join(log_dir, 'model'), overwrite=True)







