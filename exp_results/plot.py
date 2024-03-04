# plot from downloaded data
import os
import pandas as pd


root_dir = f"./data/"
print(os.path.exists(root_dir))

METHODs = ['Selfish', 'Ours', 'Ours_oldcf_cont', 'Ours-oldcf_discrete', 'team', 'Inequity']
SCENARIOs = ['Coin', 'LBF', 'Cleanup', "Harvest"]
# for each env
COLORs = ['r', 'hotpink', 'c', 'b', 'dodgerblue', 'mediumpurple',
          'cadetblue', 'steelblue', 'mediumslateblue', 'hotpink', 'mediumturquoise']

COLORs = reversed(COLORs[:len(METHODs)])

color_dict = {k: v for k, v in zip(METHODs, COLORs)}
LINE_STYPLEs = ['solid' for i in range(20)]
pos_dict = {
    'Coin': 141,
    'LBF': 142,
    'Cleanup': 143,
    'Harvest': 144,
}

# To Configure
map_scenario_to_name = {k: k for k in SCENARIOs}
map_scenario_to_name.update({})

map_method_to_name = {k: k for k in METHODs}
map_method_to_name.update({})

rewards_list = []
data_dict = {}
for scenario_tag in SCENARIOs:
    data_dict[scenario_tag] = {}
    for method_name in METHODs:
        data_dict[scenario_tag][method_name] = {}
        data_dir = os.path.join(root_dir, scenario_tag, method_name)
        if not os.path.exists(data_dir):
            print("Skip: ", data_dir)
            continue
        d_list = os.listdir(data_dir)
        # load from csv
        for d in d_list:
            if d[-4:] != ".csv":
                continue
            seed = d[:-4].split("_")[-1]
            raw_data = pd.read_csv(os.path.join(data_dir, d))
            for row_key in raw_data.keys():
                if row_key == "Step" or 'MIN' in row_key or 'MAX' in row_key:
                    continue
                rewards = raw_data[row_key]
            if scenario_tag == 'Coin':
                data_dict[scenario_tag][method_name][seed] = rewards[: 200]
            elif scenario_tag == 'Cleanup':
                data_dict[scenario_tag][method_name][seed] = rewards[60 : 157]
            elif scenario_tag == 'LBF':
                data_dict[scenario_tag][method_name][seed] = rewards[: 200]
            else:
                data_dict[scenario_tag][method_name][seed] = rewards[: 200]

import matplotlib.pyplot as plt
import numpy as np
from plot_utils import *
# plt.style.use('fivethirtyeight')
sorted_methods_list = METHODs


def draw_each(env_name, data_dict, i, color_list, map_method_to_name):
    plt.subplot(i)

    for method in sorted_methods_list:
        data = data_dict[method]
        color = color_list[method]
        seed_num = len(data.keys())
        len_list = [len(data[k_d]) for k_d in data.keys()]
        if len(len_list) == 0:
            print(f"Skip {env_name} {method}, since no data")
            continue
        max_length = np.array(len_list).max()
        # print(method, data.keys(), max_length)

        timestep = np.arange(max_length) * 6400  # / 100

        reward_plot = np.zeros((seed_num, max_length))
        for meth_name, data_i in data.items():
            if len(data_i) == max_length:
                reward_plot = np.array(list(data_i))[np.newaxis, :]
                # reward_plot = np.nan_to_num(reward_plot)
                reward_plot = reward_plot.repeat(seed_num, axis=0)
                break
        lw = 2.5

        for index, run_name in enumerate(data.keys()):
            data[run_name] = np.nan_to_num(data[run_name])
            reward_plot[index, :len(data[run_name])] = data[run_name]
        reward = np.array(reward_plot)
        r_mean, r_std = np.mean(reward, axis=0), np.std(reward, axis=0, ddof=1)
        # if config['min_value'] is not None:
        # plt.ylim(config['min_value'], config['max_value'])
        # plt.xlim(-1, config['max_step'])
        # plt.ylim(-2., 0.5)
        # else:
        # plt.set_ylim(x.min() *2, x.max()*2)

        import matplotlib as mpl
        mpl.rcParams['axes.formatter.use_mathtext'] = True
        mpl.rcParams['mathtext.fontset'] = 'stix'
        mpl.rcParams['font.size'] = 20

        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        # plt.set_yticklabels(plt.get_yticks(), fontsize=20)
        # plt.set_xticklabels(plt.get_xticks(), fontsize=20)
        # timestep = (timestep / 10000).i
        plt.plot(timestep * 100, exponential_moving_average(r_mean, 0.1), color=color,
                 label=map_method_to_name[method],  # +'-' + str(seed_num),
                 linewidth=lw, linestyle='solid',
                 )
        # set yticks
        # plt.xticks(np.arange(0, 80000, 10000))
        # print(method, r_mean)
        # r_std = r_std * 0.7
        plt.fill_between(timestep * 100, exponential_moving_average(r_mean - r_std, 0.1), exponential_moving_average(r_mean + r_std, 0.1), alpha=0.1,
                         color=color)
        # if 'causal' in config["method"]:
    # plt.legend(loc="best", bbox_to_anchor=(1.0, 0.0), borderaxespad=0.1, borderpad=0.2, fontsize=7)
    # plt.ylabel('Reward', fontsize=32)

    axes = plt.gca()
    axes.set_title(env_name, fontsize=32, y=1.07)

    plt.xlabel('Number of Timesteps', fontsize=25, loc='center')  # $(×10^4)
    plt.grid()

# plot each scenario as a subplot, and each method as a line

plt.figure(figsize=(32, 6))
# print([k for k, v in DATA.items()])
# plt.text(0.06, 0.5, "y_label", va='center', rotation='vertical', fontsize=32)
for scenario_tag, data_for_each_env in data_dict.items():
    # print(env_name)
    if scenario_tag in pos_dict:
        i = pos_dict[scenario_tag]
    else:
        print(
            f"Skipping environment {scenario_tag} because it is not in the pos_dict dictionary.")
        continue

    draw_each(map_scenario_to_name[scenario_tag],
              data_for_each_env, i, color_list=color_dict, map_method_to_name=map_method_to_name)


fig = plt.gcf()
ax = fig.get_axes()[0]
handles, labels = ax.get_legend_handles_labels()
# plt.legend(loc="best", bbox_to_anchor=(1.0, 0.0), borderaxespad=0.1, borderpad=0.2, fontsize=7)
print(len(handles), len(labels))
print(labels)
# labels = [s.split('---5')[0] for s in labels]
# labels = METHODs
fig.text(0.085, 0.5, "Social Good",
         va='center', rotation='vertical', fontsize=32)
# fig.text(0.5, 0.0, "Number of frames $(×10^4)$", ha='center', va='center', fontsize=32)

legend = fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=32,
                    bbox_to_anchor=(0.5, -0.3), bbox_transform=fig.transFigure)

for line in legend.get_lines():
    line.set_linewidth(5)
plt.subplots_adjust(hspace=0.45)


plt.savefig("main_results.pdf", bbox_inches='tight')
plt.show()