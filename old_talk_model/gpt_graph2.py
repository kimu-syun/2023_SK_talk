# グラフ描写
import gpt_talk2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as wg
import matplotlib.cm as cm


class value:
    t = 10
    a = 3
    parent_valence = 0
    parent_arousal = 0
    child_valence = 0
    child_arousal = 0


v = value()

fig_parent_obs = plt.figure(figsize=(5, 5))
fig_parent_sta = plt.figure(figsize=(5, 5))
fig_child_obs = plt.figure(figsize=(5, 5))
fig_child_sta = plt.figure(figsize=(5, 5))
fig_operater = plt.figure(figsize=(8, 3))
fig_operater.suptitle("Operater", fontsize=18)
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

# スライダー
ax_t = fig_operater.add_axes([0.2, 0.60, 0.6, 0.04])
ax_a = fig_operater.add_axes([0.2, 0.50, 0.6, 0.04])
ax_parent_val = fig_operater.add_axes([0.2, 0.40, 0.6, 0.04])
ax_parent_aro = fig_operater.add_axes([0.2, 0.30, 0.6, 0.04])
ax_child_val = fig_operater.add_axes([0.2, 0.20, 0.6, 0.04])
ax_child_aro = fig_operater.add_axes([0.2, 0.10, 0.6, 0.04])


sli_t = wg.Slider(ax_t, "time", 1, 100, valinit=v.t, valstep=1)
sli_a = wg.Slider(ax_a, "parent action", 1, 5, valinit=v.a, valstep=1)
sli_parent_val = wg.Slider(
    ax_parent_val, "parent valence", -10, 10, valinit=v.parent_valence, valstep=1
)
sli_parent_aro = wg.Slider(
    ax_parent_aro, "parent arousal", -10, 10, valinit=v.parent_arousal, valstep=1
)
sli_child_val = wg.Slider(
    ax_child_val, "child valence", -10, 10, valinit=v.child_valence, valstep=1
)
sli_child_aro = wg.Slider(
    ax_child_aro, "child arousal", -10, 10, valinit=v.child_arousal, valstep=1
)


# ボタン
ax_btn_do = fig_operater.add_axes([0.02, 0.80, 0.30, 0.1])
ax_btn_reset = fig_operater.add_axes([0.35, 0.80, 0.30, 0.1])
ax_btn_close = fig_operater.add_axes([0.68, 0.80, 0.30, 0.1])

btn_do = wg.Button(ax_btn_do, "Talk Start", color="#f8e58c", hovercolor="#38b48b")
btn_reset = wg.Button(ax_btn_reset, "Reset", color="crimson", hovercolor="magenta")
btn_close = wg.Button(ax_btn_close, "Close", color="aqua", hovercolor="lime")


# 変数変更
def update(val):
    v.t = sli_t.val
    v.a = sli_a.val
    v.parent_valence = sli_parent_val.val
    v.parent_arousal = sli_parent_aro.val
    v.child_valence = sli_child_val.val
    v.child_arousal = sli_child_aro.val


# 実行ボタン
def btn_do_click(event):
    init_data = np.array(
        {
            "time": v.t,
            "act": v.a,
            "parent_val": v.parent_valence,
            "parent_aro": v.parent_arousal,
            "child_val": v.child_valence,
            "child_aro": v.child_arousal,
        }
    )
    np.save("./data2/init_data", init_data)

    data_name = ["parent_obs", "parent_state", "child_obs", "child_state"]
    # モデル実行
    all_history = gpt_talk2.main(
        v.t,
        v.a - 1,
        [v.parent_valence, v.parent_arousal],
        [v.child_valence, v.child_arousal],
    )

    for i, ax in enumerate(graphs):
        plt.figure(i + 1)
        plt.axes(ax)
        graph_write(ax, i)
        history = np.vstack(all_history[i]).astype(float)
        temporal_colormap = cm.hot(np.linspace(0, 1, v.t))
        ax.plot(history[:, 0], history[:, 1], "r", zorder=1)
        ax.scatter(
            history[:, 0],
            history[:, 1],
            c=temporal_colormap,
            s=30,
            zorder=2,
            ec="k",
        )
        np.save(f"./data2/{data_name[i]}", history)
        plt.draw()


# リセットボタン
def btn_reset_click(event):
    for i, ax in enumerate(graphs):
        plt.figure(i + 1)
        plt.axes(ax)
        plt.gca().clear()


# クローズボタン
def btn_close_click(event):
    for i in range(5):
        plt.figure(i + 1)
        plt.close()


sli_t.on_changed(update)
sli_a.on_changed(update)
sli_parent_aro.on_changed(update)
sli_parent_val.on_changed(update)
sli_child_aro.on_changed(update)
sli_child_val.on_changed(update)

btn_do.on_clicked(btn_do_click)
btn_reset.on_clicked(btn_reset_click)
btn_close.on_clicked(btn_close_click)

plt.show()
