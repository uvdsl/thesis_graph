import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if not os.path.exists(f"./img/"):
        os.mkdir(f"./img/")

folder = f"./data/"
file_names = ["c_al_n","c_al_nn","c_sl_n","c_sl_nn","c_bl_n","c_bl_nn","t_al_n","t_al_nn","t_sl_n","t_sl_nn","t_bl_n","t_bl_nn"]
df = pd.DataFrame()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

for fn in file_names:
    data = pd.read_pickle(f"{folder}{fn}.pkl")
    data['label'] = ['baseline', 'discon.', 'recon.' , 'extended']
    stats = []
    for ix in range(data.shape[0]):
        stats.append({
            "label": data.loc[ix, 'label'],  # not required
            # "mean":  5,  # not required
            "med": data.loc[ix,'median'],
            "q1": data.loc[ix,'lower_quartile'],
            "q3": data.loc[ix,'upper_quartile'],
            # "cilo": 5.3 # not required
            # "cihi": 5.7 # not required
            "whislo": data.loc[ix,'lower_whisker'],  # required
            "whishi": data.loc[ix,'upper_whisker'],  # required
            "fliers": []  # required if showfliers=True
            })

 

    fig, axes = plt.subplots()
    if fn[-2:] == '_n':
        axes.set_ylim(-0.00025,  0.0105) # 0.0235) #
        axes.set_yticks([0, 0.001, 0.002, 0.003, 0.004,
                       0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 
                    #    0.011, 0.012, 0.013, 0.014,
                    #    0.015, 0.016, 0.017, 0.018, 0.019, 0.020, 0.021, 0.022, 0.023 
                       ])
        axes.set_title('Normalized', fontsize=18)
    else:
        axes.set_title('Non-Normalized', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=14)
    axes.bxp(stats)
    # axes.set_title(fn, fontsize=20)
    plt.savefig(f'./img/{fn}.pdf')
# plt.show()
