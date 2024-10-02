import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plot_utils import exponential_moving_average

root_dir = f"./data/"
print(os.path.exists(root_dir))

# METHODs = ['Selfish', 'Inequity', 'SVO', 'CF']
# SCENARIOs = ['Coin_3_Agents', 'Coin_4_Agents','LBF_3_Agents',  'LBF_4_Agents']
# COLORs = ['r', 'hotpink', 'c', 'b', 'dodgerblue', 'mediumpurple',
#           'cadetblue', 'steelblue', 'mediumslateblue', 'hotpink', 'mediumturquoise']
# COLORs = list(reversed(COLORs[:len(METHODs)]))
# color_dict = {k: v for k, v in zip(METHODs, COLORs)}
# LINE_STYPLEs = ['solid' for _ in range(20)]
# pos_dict = {
#     'Coin_3_Agents': 141,
#     'Coin_4_Agents': 142,
#     'LBF_3_Agents': 143,
#     'LBF_4_Agents': 144,
# }


METHODs = ['Selfish', 'Inequity', 'SVO', 'CF']
SCENARIOs = ['Common_Harvest_5', 'Common_Harvest_7', 'Cleanup_5', 'Cleanup_7']
COLORs = ['r', 'hotpink', 'c', 'b', 'dodgerblue', 'mediumpurple',
          'cadetblue', 'steelblue', 'mediumslateblue', 'hotpink', 'mediumturquoise']
COLORs = list(reversed(COLORs[:len(METHODs)]))
color_dict = {k: v for k, v in zip(METHODs, COLORs)}
LINE_STYPLEs = ['solid' for _ in range(20)]
pos_dict = {
    'Common_Harvest_5': 141,
    'Common_Harvest_7': 142,
    'Cleanup_5': 143,
    'Cleanup_7': 144,
}



# To Configure
map_scenario_to_name = {k: k for k in SCENARIOs}
map_method_to_name = {f'Selfish': fr'Selfish', f'Inequity': fr'Inequity', f'SVO': fr'SVO', f'CF': fr'CF(Ours)'}

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
            if scenario_tag == 'Common_Harvest_5' or scenario_tag == 'Common_Harvest_7' or scenario_tag == 'Cleanup_5' or scenario_tag == 'Cleanup_7':
                data_dict[scenario_tag][method_name][seed] = rewards
            else:
                data_dict[scenario_tag][method_name][seed] = rewards

sorted_methods_list = METHODs


def draw_each(env_name, data_dict, i, color_list, map_method_to_name, label):
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
        timestep = np.arange(max_length) * 6400  # Adjusting the scale here

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

        # if method == 'CF':
        #     # 使用移动平均来平滑数据
        #     window = 50  # 调整窗口大小以获得适当的平滑效果
        #     r_mean_smooth = np.convolve(r_mean, np.ones(window)/window, mode='valid')
        #     r_std_smooth = np.convolve(r_std, np.ones(window)/window, mode='valid')
        #     timestep_smooth = timestep[window-1:]

        #     plt.plot(timestep_smooth / 1e8, r_mean_smooth, color=color,
        #             label=map_method_to_name[method],
        #             linewidth=2.5, linestyle='solid')
            
        #     # 使用 fill_between 绘制平滑后的偏差带
        #     plt.fill_between(timestep_smooth / 1e8, 
        #                     r_mean_smooth - r_std_smooth, 
        #                     r_mean_smooth + r_std_smooth, 
        #                     alpha=0.1, color=color)


        import matplotlib as mpl
        mpl.rcParams['axes.formatter.use_mathtext'] = True
        mpl.rcParams['mathtext.fontset'] = 'stix'
        mpl.rcParams['font.size'] = 20

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        # Select range for "Cleanup" environment

        plt.plot(timestep / 1e8, exponential_moving_average(r_mean, 0.1), color=color,
                label=map_method_to_name[method],
                linewidth=lw, linestyle='solid',
                )
        plt.fill_between(timestep / 1e8, exponential_moving_average(r_mean - r_std, 0.1), exponential_moving_average(r_mean + r_std, 0.1), alpha=0.1,
                        color=color)

    axes = plt.gca()
    axes.set_title(f"{label} {env_name}", fontsize=32, y=1.07)

    plt.xlabel('Number of Timesteps (×1e7)', fontsize=25, loc='center')
    plt.grid()

    if env_name == 'Cleanup_5' or env_name == 'Cleanup_7':
        plt.ylim(top=200)
        plt.ylim(bottom=-50)
    elif env_name == 'Common_Harvest_5':
        plt.ylim(top=1000)
        plt.ylim(bottom=-50)
    elif env_name == 'Common_Harvest_7':
        plt.ylim(top=800)
        plt.ylim(bottom=-50)
    else:
        plt.ylim(bottom=-50)

    # Setting the x-axis scale to match 10^8
    x_ticks = np.linspace(0, max(timestep) / 1e8, num=6)
    x_ticks_labels = ['0', '0.4', '0.8', '1.2', '1.6', '2.0']
    plt.xticks(ticks=x_ticks, labels=x_ticks_labels)


plt.figure(figsize=(32, 6))
labels = ['(a)', '(b)', '(c)', '(d)']
for idx, (scenario_tag, data_for_each_env) in enumerate(data_dict.items()):
    if scenario_tag in pos_dict:
        i = pos_dict[scenario_tag]
    else:
        print(
            f"Skipping environment {scenario_tag} because it is not in the pos_dict dictionary.")
        continue

    draw_each(map_scenario_to_name[scenario_tag],
              data_for_each_env, i, color_list=color_dict, map_method_to_name=map_method_to_name, label=labels[idx])

fig = plt.gcf()
ax = fig.get_axes()[0]
handles, labels = ax.get_legend_handles_labels()
print(len(handles), len(labels))
print(labels)
fig.text(0.085, 0.5, "Collective Reward",
         va='center', rotation='vertical', fontsize=32)

legend = fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=32,
                    bbox_to_anchor=(0.5, -0.3), bbox_transform=fig.transFigure)

for line in legend.get_lines():
    line.set_linewidth(5)


plt.subplots_adjust(hspace=0.45)

plt.savefig("main_results_hard.pdf", bbox_inches='tight')
plt.show()
