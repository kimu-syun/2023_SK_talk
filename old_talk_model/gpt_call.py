import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as wg
import matplotlib.cm as cm

fig_parent_obs = plt.figure(figsize=(5, 5))
fig_parent_sta = plt.figure(figsize=(5, 5))
fig_child_obs = plt.figure(figsize=(5, 5))
fig_child_sta = plt.figure(figsize=(5, 5))
ax_parent_obs = fig_parent_obs.add_subplot(1, 1, 1)
ax_parent_sta = fig_parent_sta.add_subplot(1, 1, 1)
ax_child_obs = fig_child_obs.add_subplot(1, 1, 1)
ax_child_sta = fig_child_sta.add_subplot(1, 1, 1)

graphs = (ax_parent_obs, ax_parent_sta, ax_child_obs, ax_child_sta)

ax_title = [
    "Parent observation",
    "Parent hidden state",
    "Child observation",
    "Child hidden state",
]


def graph_write(ax, i):
    # グラフ範囲
    ax.set_xlim([-11, 11])
    ax.set_ylim([-11, 11])
    # グラフ目盛り
    ax.set(
        title=ax_title[i],
        xlabel="valence parameter",
        ylabel="arousal parameter",
        xticks=[-10, -5, 0, 5, 10],
        yticks=[-10, -5, 0, 5, 10],
    )
    ax.minorticks_on()
    ax.grid(which="major", axis="both", linestyle="-")
    ax.grid(which="minor", axis="both", linestyle="--")
    ax.axhline(c="k")
    ax.axvline(c="k")
    ax.set_xlabel("valence parameter")
    ax.set_ylabel("arousal parameter")


for i, ax in enumerate(graphs):
    graph_write(ax, i)

# 実行ボタン
init_data = np.load("./data/init_data.npy", allow_pickle=True).item()
data_name = ["parent_obs", "parent_state", "child_obs", "child_state"]
# データを取り出してグラフ描画
for i, ax in enumerate(graphs):
    history = np.load(f"./data/{data_name[i]}.npy")
    plt.figure(i + 1)
    plt.axes(ax)
    graph_write(ax, i)
    temporal_colormap = cm.hot(np.linspace(0, 1, init_data['time']))
    ax.plot(history[:, 0], history[:, 1], "r", zorder=1)
    ax.scatter(
        history[:, 0],
        history[:, 1],
        c=temporal_colormap,
        s=30,
        zorder=2,
        ec="k",
    )
    plt.draw()

plt.show()
