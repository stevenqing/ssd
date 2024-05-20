import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plot_utils import exponential_moving_average

root_dir = f"./data/"
print(os.path.exists(root_dir))

METHODs = ['Selfish', 'Inequity', 'SVO', 'CF_2']
METHOD_2 = ['CF_001', 'CF_01', 'CF_1', 'CF_10', 'CF_2']
TOTAL_METHODS = ['Selfish', 'Inequity', 'SVO', 'CF_001', 'CF_01', 'CF_1', 'CF_2', 'CF_10']
SCENARIOs = ['Coin4', 'Coin5', 'Coin4_Ablation', 'Coin5_Ablation']
COLORs = ['r', 'hotpink', 'c', 'b']
COLORs = list(reversed(COLORs[:len(METHODs)]))
color_dict = {k: v for k, v in zip(METHODs, COLORs)}

COLOR_2 = ['r', "indianred", "darkred", "salmon", "lightcoral"]
COLOR_2 = list(reversed(COLOR_2[:len(METHOD_2)]))
color_dict_2 = {k: v for k, v in zip(METHOD_2, COLOR_2)}

LINE_STYPLEs = ['solid' for i in range(20)]
pos_dict = {
    'Coin4': 141,
    'Coin5': 142,
    'Coin4_Ablation': 143,
    'Coin5_Ablation': 144,
}

map_scenario_to_name = {k: k for k in SCENARIOs}
map_method_to_name = {k: k for k in METHODs}
map_method_to_name_2 = {k: k for k in METHOD_2}

rewards_list = []
data_dict = {}
for scenario_tag in SCENARIOs:
    data_dict[scenario_tag] = {}
    for method_name in TOTAL_METHODS:
        data_dict[scenario_tag][method_name] = {}
        data_dir = os.path.join(root_dir, scenario_tag, method_name)
        if not os.path.exists(data_dir):
            print("Skip: ", data_dir)
            continue
        d_list = os.listdir(data_dir)
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
                data_dict[scenario_tag][method_name][seed] = rewards[:158]
            elif scenario_tag == 'Cleanup':
                data_dict[scenario_tag][method_name][seed] = rewards[30:158]
            elif scenario_tag == 'LBF':
                data_dict[scenario_tag][method_name][seed] = rewards[:158]
            else:
                data_dict[scenario_tag][method_name][seed] = rewards[:158]


sorted_methods_list = METHODs
sorted_methods_list_2 = METHOD_2

def draw_each(env_name, data_dict, i, color_list, color_list_2, map_method_to_name, map_method_to_name_2):
    plt.subplot(i)

    if i in [141, 142]:
        methods_list = sorted_methods_list
        color_list = color_list
        map_method_to_name = map_method_to_name
    else:
        methods_list = sorted_methods_list_2
        color_list = color_list_2
        map_method_to_name = map_method_to_name_2

    for method in methods_list:
        data = data_dict[method]
        color = color_list[method]
        seed_num = len(data.keys())
        len_list = [len(data[k_d]) for k_d in data.keys()]
        if len(len_list) == 0:
            print(f"Skip {env_name} {method}, since no data")
            continue
        max_length = np.array(len_list).max()
        timestep = np.arange(max_length) * 6400

        reward_plot = np.zeros((seed_num, max_length))
        for meth_name, data_i in data.items():
            if len(data_i) == max_length:
                reward_plot = np.array(list(data_i))[np.newaxis, :]
                reward_plot = reward_plot.repeat(seed_num, axis=0)
                break
        lw = 2.5

        for index, run_name in enumerate(data.keys()):
            data[run_name] = np.nan_to_num(data[run_name])
            reward_plot[index, :len(data[run_name])] = data[run_name]
        reward = np.array(reward_plot)
        r_mean, r_std = np.mean(reward, axis=0), np.std(reward, axis=0, ddof=1)

        import matplotlib as mpl
        mpl.rcParams['axes.formatter.use_mathtext'] = True
        mpl.rcParams['mathtext.fontset'] = 'stix'
        mpl.rcParams['font.size'] = 20

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        if env_name == 'Cleanup':
            timestep = timestep[30:158]
            r_mean = r_mean[30:158]
            r_std = r_std[30:158]

        plt.plot(timestep / 1e8, exponential_moving_average(r_mean, 1), color=color,
                 label=map_method_to_name[method],
                 linewidth=lw, linestyle='solid',
                 )
        plt.fill_between(timestep / 1e8, exponential_moving_average(r_mean - r_std, 1), exponential_moving_average(r_mean + r_std, 0.1), alpha=0.1,
                         color=color)

    # 添加所有方法到图例，即使没有数据
    for method in methods_list:
        if method not in data_dict or len(data_dict[method]) == 0:
            plt.plot([], [], color=color_list[method], label=map_method_to_name[method])

    axes = plt.gca()
    axes.set_title(env_name, fontsize=32, y=1.07)

    plt.xlabel('Number of Timesteps (×1e7)', fontsize=25, loc='center')
    plt.grid()

    x_ticks = np.linspace(0, max(timestep) / 1e8, num=6)
    x_ticks_labels = ['0', '0.2', '0.4', '0.6', '0.8', '1.0']
    plt.xticks(ticks=x_ticks, labels=x_ticks_labels)

plt.figure(figsize=(32, 6))
for scenario_tag, data_for_each_env in data_dict.items():
    if scenario_tag in pos_dict:
        i = pos_dict[scenario_tag]
    else:
        print(
            f"Skipping environment {scenario_tag} because it is not in the pos_dict dictionary.")
        continue

    draw_each(map_scenario_to_name[scenario_tag],
              data_for_each_env, i, color_list=color_dict, color_list_2=color_dict_2, map_method_to_name=map_method_to_name, map_method_to_name_2=map_method_to_name_2)

fig = plt.gcf()
ax = fig.get_axes()[0]
handles, labels = ax.get_legend_handles_labels()
print(len(handles), len(labels))  # 打印图例条目的数量和标签
print(labels)
fig.text(0.085, 0.5, "Collective Reward",
         va='center', rotation='vertical', fontsize=32)

legend = fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=32,
                    bbox_to_anchor=(0.5, -0.3), bbox_transform=fig.transFigure)

for line in legend.get_lines():
    line.set_linewidth(5)  # 将线条宽度设置为5

ax = fig.get_axes()[2]
handles, labels = ax.get_legend_handles_labels()
print(len(handles), len(labels))
print(labels)

legend = fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=32,
                    bbox_to_anchor=(0.5, -0.5), bbox_transform=fig.transFigure)

for line in legend.get_lines():
    line.set_linewidth(5)


plt.subplots_adjust(hspace=0.45)

plt.savefig("coin_variant.pdf", bbox_inches='tight')
plt.show()
