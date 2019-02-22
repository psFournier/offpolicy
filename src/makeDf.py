import glob
import pandas as pd
import os

DIR = '../log/cluster/2102Eps/'
ENV = '*-v0'
runs = glob.glob(os.path.join(DIR, ENV, '*'))
frames = []

for run in runs:

    config = pd.read_json(os.path.join(run, 'config.txt'), lines=True)
    try:
        df = pd.read_json(os.path.join(run, 'progress.json'), lines=True)
        config = pd.concat([config] * df.shape[0], ignore_index=True)
        data = pd.concat([df, config], axis=1)
        data['num_run'] = run.split('/')[5]
        frames.append(data)
    except:
        print(run, 'not ok')
df = pd.concat(frames, ignore_index=True)
df.to_pickle(os.path.join(DIR, ENV + '.pkl'))