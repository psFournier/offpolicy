import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import brewer2mpl
import matplotlib.ticker as ticker
from matplotlib import rc, rcParams
from matplotlib import lines

# rc('font',family='serif')
# rc('text', usetex=True)
# rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"
#                                  r"\usepackage{amssymb}"]
bmap = brewer2mpl.get_map('set1', 'qualitative', 9)
colors = bmap.mpl_colors

def quant_inf(x):
    return x.quantile(0.25)
def quant_sup(x):
    return x.quantile(0.75)

dirs = ['1502']
df = pd.concat([pd.read_pickle('../log/cluster/{}/*-v0.pkl'.format(d)) for d in dirs], ignore_index=True)

x = ['step']
params = ['--agent',
          '--batchsize',
          '--env',
          '--eval_freq',
          '--gamma',
          '--ep_tasks',
          '--rnd_demo',
          '--wimit',
          '--ep_steps',
          '--inv_grad',
          '--margin',
          '--tutoronly',
          '--demo',
          '--network',
          '--prop_demo',
          '--freq_demo',
          '--filter',
          '--lrimit',
          '--rndv',
          '--initq',
          '--layers',
          '--her',
          '--nstep',
          '--alpha'
          ]

a, b = 1,1
fig, axes = plt.subplots(a, b, figsize=(15,9), squeeze=False, sharex=True)

df1 = df.copy()
df1 = df1[(df1['--env'] == 'Rooms9-v0')]
# df1 = df1[(df1['--her'] == 4)]
# df1 = df1[(df1['--nstep'] == 4)]

y = 'term'
for p in params: print(p, df1[p].unique())
df1 = df1.groupby(x + params).agg({y:[np.median, np.mean, np.std, quant_inf, quant_sup]}).reset_index()
p1 = [p for p in params if len(df1[p].unique()) > 1]
# p1 = 'num_run'

for j, (name, g) in enumerate(df1.groupby(p1)):
    axes[0, 0].plot(g['step'], g[y]['median'], label=name)
    axes[0, 0].fill_between(g['step'],
                           g[y]['quant_inf'],
                           g[y]['quant_sup'], alpha=0.25, linewidth=0)
    # axes[0, 0].scatter(g['step'], g[y], label=name)
axes[0, 0].legend()
# axes[0, 0].set_xlim([0,100000])

plt.show()