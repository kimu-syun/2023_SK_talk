# 対話モデル
# 学習なし、感情遷移なし

import numpy as np
from pymdp.agent import Agent
from pymdp import utils, maths
import openai
import os
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
from scipy import stats
import global_value as g

from gpt_child_act import chi_text_gpt
from gpt_child_obs import chi_obs_gpt
from gpt_parent_act import par_text_gpt

# g.gpt_model = "gpt-3.5-turbo"
g.gpt_model = "gpt-4"

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")


# 環境設定
x = 5
y = 5

grid_dims = [2*x + 1, 2*y + 1]
num_grid_points = np.prod(grid_dims)

emo_list = []
for i in range(-x, x + 1):
    for j in range(-y, y + 1):
        emo_list.append((i, j))

action_name = (
    "話を逸らす",
    "協力を求める",
    "時間を交渉する",
    "反抗する",
    "無視する",
)

# number of variable elements
num_states = [num_grid_points]
num_obs = [num_grid_points]
num_contorols = [len(action_name)]

# 生成モデル

# 尤度(p(o|s))
# likelihood.pyで生成したデータを使用
A_m_shapes = [[o_dim] + num_states for o_dim in num_obs]
A = utils.obj_array_ones(A_m_shapes)
for s in emo_list:
    pos = np.load(Rf"A\dis\{list(s)}.npy")
    for o in emo_list:
        A[0][emo_list.index(o), emo_list.index(s)] = pos[5-o[1], o[0]+5]
A = utils.norm_dist_obj_arr(A)


# 遷移分布(p(s|s,u))
B_f_shapes = [[ns, ns, num_contorols[f]] for f, ns in enumerate(num_states)]
B = utils.obj_array_ones(B_f_shapes)
B = utils.norm_dist_obj_arr(B)


# 選好分布(p(o))
C = utils.obj_array_zeros(num_obs)


# 事前分布(p(s))
D = utils.obj_array_ones(num_states)
D = utils.norm_dist_obj_arr(D)

# Childクラス(agent)指定
class Child:
    def __init__(self):
        self.x = [0, 0]
        self.obs = [0, 0]
        self.a = 0
        self.text = ""
        self.log_obs = []
        self.log_act = []

        self.history_obs = []
        self.history_state = []

child = Child()

# 外部環境の設定
# アクションを受け取ってセリフを返す
class Parent():
    def __init__(self, parent_emo):
        self.emo = parent_emo
        self.text = par_text_gpt(self.emo, first=True)

    # 返答
    def step(self, text):
        self.text = par_text_gpt(text, self.emo)
        return self.text


# グラフ描画
# 真の感情と推測隠れ状態についてのグラフ
def qs_graph(qs, T, s):
    for t in range(0, T):
        grid = np.empty((grid_dims[0], grid_dims[1]))
        for aro, val in emo_list:
            value = qs[t][0][emo_list.index((aro, val))]
            grid[-val + 5][aro + 5] = value.copy()
            # if aro == 3 and val == -1:
            #     grid[-val + 5][aro + 5] = 1
        fig_qs_state = plt.figure()
        x_grid = [-5, "", "", "", "", 0, "", "", "", "", 5]
        y_grid = [5, "", "", "", "", 0, "", "", "", "", -5]
        fig_qs_state = sns.heatmap(data=grid, square=True, cmap='cool', xticklabels=x_grid, yticklabels=y_grid, linewidths=0.5)
        fig_qs_state.set_title(f"qs parent state  {t+1}times")
        fig_qs_state = plt.plot(s[0]+5.5, 5.5-s[1], marker=".", color="black", markersize=15)
        fig_qs_state = plt.savefig(Rf".\qs_graph\qs_state_{t+1}times.png")
        fig_qs_state = plt.close()

# 選好分布と観測信号のグラフ
def goal_graph(pre, T, o):
    grid = np.empty((grid_dims[0], grid_dims[1]))
    for aro, val in emo_list:
        value = pre[emo_list.index((aro, val))]
        grid[-val + 5][aro + 5] = value.copy()
    for t in range(0, T):
        fig_qs_state = plt.figure()
        x_grid = [-5, "", "", "", "", 0, "", "", "", "", 5]
        y_grid = [5, "", "", "", "", 0, "", "", "", "", -5]
        fig_qs_state = sns.heatmap(data=grid, square=True, cmap='cool', xticklabels=x_grid, yticklabels=y_grid, linewidths=0.5)
        fig_qs_state.set_title(f"obs_histry  {t+1}times")
        # fig_qs_state = plt.plot(o[0][0:t+1]+5.5, -o[1][0:t+1] + 5.5, marker=".", color="black", markersize=15)
        fig_qs_state = plt.plot(o[0][t]+5.5, -o[1][t] + 5.5, marker=".", color="black", markersize=15)
        fig_qs_state = plt.savefig(Rf".\obs\obshis_{t+1}times.png")
        fig_qs_state = plt.close()


# 対話モデル
def main(T, par_emo, pre_emo):
    # 選好分布作成(p(o))
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if (-x <= pre_emo[0] + dx <= x) and (-y <= pre_emo[1] + dy <= y):
                C[0][emo_list.index(tuple([pre_emo[0]+dx, pre_emo[1]+dy]))] = 1
    C[0][emo_list.index(tuple(pre_emo))] = 3

    my_agent = Agent(
        A=A,
        B=B,
        C=C,
        D=D,
        policy_len=2,
        save_belief_hist=True,
        )
    my_env = Parent(parent_emo=par_emo)
    obs_his = np.empty((2, T))

    print(f"First parent : {my_env.text}")

    for t in range(T):
        print(f"{t+1} time")
        # 子供の応答生成
        child.text = chi_text_gpt(action_name[child.a], my_env.text)
        print(f"child : {child.text}")
        # 親の発言生成
        my_env.text = my_env.step(child.text)
        print(f"parent : {my_env.text}")
        # 子供の感情受け取り
        child.obs = [emo_list.index(tuple(chi_obs_gpt(my_env.text)))]
        obs_his[0][t] = emo_list[child.obs[0]][0]
        obs_his[1][t] = emo_list[child.obs[0]][1]
        print(f"child obs : {emo_list[child.obs[0]]}")
        # 能動推論
        # 隠れ状態の推論
        qs = my_agent.infer_states(child.obs)
        # 行動選択
        qpi, G = my_agent.infer_policies()
        child.a = int(my_agent.sample_action()[0])
        print(f"child next act : {action_name[child.a]}")
        print("------------------------------------------------")

    # グラフ作成
    shutil.rmtree(R".\qs_graph")
    shutil.rmtree(R".\obs")
    os.mkdir(R".\qs_graph")
    os.mkdir(R".\obs")
    qs_graph(my_agent.qs_hist, T=T, s=par_emo)

    goal_graph(my_agent.C[0], T=T, o=obs_his)

    print("end")

# 回す回数、親の隠れ状態、選好
step = int(input(f"何回回す？ : "))
hid_s = list(map(int, input(f"親の感情は？ : ").split()))
pre_o = list(map(int, input(f"子供の選好は？ : ").split()))
main(step, hid_s, pre_o)
