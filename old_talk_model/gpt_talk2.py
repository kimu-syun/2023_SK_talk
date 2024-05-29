# gptを用いた対話モデル（メインファイル）
import numpy as np
from pymdp.agent import Agent
from pymdp import utils, maths
import openai
import os
from dotenv import load_dotenv
import random
import math
import copy
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
from scipy import stats

# gpt関連のファイル読み込み
from gpt_child_act2 import child_act  # x(list), a(int) -> text(str)
from gpt_child_obs2 import child_obs  # log(list), text(str) -> x(list), log(list)
from gpt_parent_act2 import parent_act  # x(list), a(int) -> text(str)
from gpt_parent_obs2 import parent_obs  # log(list), text(str) -> x(list), log(list)

import global_value as g

g.gpt_model = "gpt-3.5-turbo"
# g.gpt_model = "gpt-4"


class Agent_variable:
    def __init__(self):
        self.x = [0, 0]
        self.y = [0, 0]
        self.a = ""
        self.text = ""
        self.log_obs = []
        self.log_act = []

        self.history_obs = []
        self.history_state = []



# APIキーをセット
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

x_limit = 10
y_limit = 10

# emotion grid
grid_dims = [2*x_limit+1, 2*y_limit+1]
num_grid_points = np.prod(grid_dims)

emo_list = []
for i in range(-x_limit, x_limit+1):
    for j in range(-y_limit, y_limit+1):
        emo_list.append((i, j))

# action name
# parent_action_name = ["praise", "nutral", "scold"]
child_action_name = ["掃除しない", "掃除を後回しにする", "掃除する"]
index_emo = ["fear", "anger", "despair", "sad", "depress", "bore", "sleep", "relax", "satisfy", "happy", "pride", "surprise"]

# number of variable elements
num_states = [num_grid_points]
num_obs = [num_grid_points, num_grid_points]
num_contorols = [len(child_action_name)]

# general model(Child)
# p(o|s)
A_m_shapes = [[o_dim] + num_states for o_dim in num_obs]
A = utils.obj_array_zeros(A_m_shapes)

# p(o1|s1)
# s1での場合分け
for parent_emotion in emo_list:
    x = parent_emotion[0]
    y = parent_emotion[1]
    if x == -x_limit or x == x_limit or y == -y_limit or y == y_limit:
        # s1が四隅
        if parent_emotion in [(-x_limit, -y_limit), (-x_limit, y_limit), (x_limit, -y_limit), (x_limit, y_limit)]:
            if parent_emotion == (-x_limit, -y_limit):
                    A[0][emo_list.index((-x_limit, -y_limit)), emo_list.index(parent_emotion)] = 0.4
                    A[0][emo_list.index((-x_limit, -y_limit+1)), emo_list.index(parent_emotion)] = 0.2
                    A[0][emo_list.index((-x_limit+1, -y_limit+1)), emo_list.index(parent_emotion)] = 0.2
                    A[0][emo_list.index((-x_limit+1, -y_limit)), emo_list.index(parent_emotion)] = 0.2
            if parent_emotion == (-x_limit, y_limit):
                    A[0][emo_list.index((-x_limit, y_limit)), emo_list.index(parent_emotion)] = 0.4
                    A[0][emo_list.index((-x_limit+1, y_limit)), emo_list.index(parent_emotion)] = 0.2
                    A[0][emo_list.index((-x_limit+1, y_limit-1)), emo_list.index(parent_emotion)] = 0.2
                    A[0][emo_list.index((-x_limit, y_limit-1)), emo_list.index(parent_emotion)] = 0.2
            if parent_emotion == (x_limit, -y_limit):
                    A[0][emo_list.index((x_limit, -y_limit)), emo_list.index(parent_emotion)] = 0.4
                    A[0][emo_list.index((x_limit-1, -y_limit)), emo_list.index(parent_emotion)] = 0.2
                    A[0][emo_list.index((x_limit-1, -y_limit+1)), emo_list.index(parent_emotion)] = 0.2
                    A[0][emo_list.index((x_limit, -y_limit+1)), emo_list.index(parent_emotion)] = 0.2
            if parent_emotion == (x_limit, y_limit):
                    A[0][emo_list.index((x_limit, y_limit)),  emo_list.index(parent_emotion)] = 0.4
                    A[0][emo_list.index((x_limit-1, y_limit)),  emo_list.index(parent_emotion)] = 0.2
                    A[0][emo_list.index((x_limit-1, y_limit-1)),  emo_list.index(parent_emotion)] = 0.2
                    A[0][emo_list.index((x_limit, y_limit-1)),  emo_list.index(parent_emotion)] = 0.2
        else:
            # s1が端っこ
            A[0][emo_list.index(parent_emotion), emo_list.index(parent_emotion)] = 0.30
            if x == -x_limit:
                A[0][emo_list.index((x, y+1)), emo_list.index(parent_emotion)] = 0.14
                A[0][emo_list.index((x, y-1)), emo_list.index(parent_emotion)] = 0.14
                A[0][emo_list.index((x+1, y+1)), emo_list.index(parent_emotion)] = 0.14
                A[0][emo_list.index((x+1, y)), emo_list.index(parent_emotion)] = 0.14
                A[0][emo_list.index((x+1, y-1)), emo_list.index(parent_emotion)] = 0.14
            if x == x_limit:
                A[0][emo_list.index((x, y+1)), emo_list.index(parent_emotion)] = 0.14
                A[0][emo_list.index((x, y-1)), emo_list.index(parent_emotion)] = 0.14
                A[0][emo_list.index((x-1, y+1)), emo_list.index(parent_emotion)] = 0.14
                A[0][emo_list.index((x-1, y)), emo_list.index(parent_emotion)] = 0.14
                A[0][emo_list.index((x-1, y-1)), emo_list.index(parent_emotion)] = 0.14
            if y == -x_limit:
                A[0][emo_list.index((x-1, y)), emo_list.index(parent_emotion)] = 0.14
                A[0][emo_list.index((x+1, y)), emo_list.index(parent_emotion)] = 0.14
                A[0][emo_list.index((x-1, y+1)), emo_list.index(parent_emotion)] = 0.14
                A[0][emo_list.index((x, y+1)), emo_list.index(parent_emotion)] = 0.14
                A[0][emo_list.index((x+1, y+1)), emo_list.index(parent_emotion)] = 0.14
            if y == x_limit:
                A[0][emo_list.index((x-1, y)), emo_list.index(parent_emotion)] = 0.14
                A[0][emo_list.index((x+1, y)), emo_list.index(parent_emotion)] = 0.14
                A[0][emo_list.index((x-1, y-1)), emo_list.index(parent_emotion)] = 0.14
                A[0][emo_list.index((x, y-1)), emo_list.index(parent_emotion)] = 0.14
                A[0][emo_list.index((x+1, y-1)), emo_list.index(parent_emotion)] = 0.14
    else:
        # s1の周りがちゃんと9個
        A[0][emo_list.index(parent_emotion), emo_list.index(parent_emotion)] = 0.2
        A[0][emo_list.index((x-1, y-1)), emo_list.index(parent_emotion)] = 0.1
        A[0][emo_list.index((x-1, y)), emo_list.index(parent_emotion)] = 0.1
        A[0][emo_list.index((x-1, y+1)), emo_list.index(parent_emotion)] = 0.1
        A[0][emo_list.index((x, y-1)), emo_list.index(parent_emotion)] = 0.1
        A[0][emo_list.index((x, y+1)), emo_list.index(parent_emotion)] = 0.1
        A[0][emo_list.index((x+1, y-1)), emo_list.index(parent_emotion)] = 0.1
        A[0][emo_list.index((x+1, y)), emo_list.index(parent_emotion)] = 0.1
        A[0][emo_list.index((x+1, y+1)), emo_list.index(parent_emotion)] = 0.1

# p(o2|s1)
A[1][:, :] = 1
A = utils.norm_dist_obj_arr(A)


# p(s|s,u)
B_f_shapes = [[ns, ns, num_contorols[f]] for f, ns in enumerate(num_states)]
B = utils.obj_array_zeros(B_f_shapes)

# https://ipsj.ixsq.nii.ac.jp/ej/?action=repository_action_common_download&item_id=188710&item_no=1&attribute_id=1&file_no=1
speak_emo = {
    "fear":(-4*(x_limit/10), 8*(y_limit/10)),
    "anger":(-7*(x_limit/10), 6*(y_limit/10)),
    "despair":(-9*(x_limit/10), 1*(y_limit/10)),
    "sad":(-8.5*(x_limit/10), -1*(y_limit/10)),
    "depress":(-6.5*(x_limit/10), -8*(y_limit/10)),
    "bore":(-4*(x_limit/10), -8.5*(y_limit/10)),
    "sleep":(1*(x_limit/10), -9.5*(y_limit/10)),
    "relax":(6.5*(x_limit/10), -6.5*(y_limit/10)),
    "satisfy":(7.5*(x_limit/10), -1*(y_limit/10)),
    "happy":(7.5*(x_limit/10), 1*(y_limit/10)),
    "pride":(5*(x_limit/10), 7.5*(y_limit/10)),
    "surprise":(0.5*(x_limit/10), 10*(y_limit/10))
    }

# p(s1|s1, a1)
# gauss = stats.norm.rvs(loc=-(child_action_name.index(a)-2)/2, scale=1.5, size=1)[0]
for parent_emotion in emo_list:
    x = parent_emotion[0]
    y = parent_emotion[1]
    for a in range(len(child_action_name)):
        locx = 0
        locy = 0
        if "掃除しない" == child_action_name[a]:
            locx = x-1
            locy = y+1
        if "掃除を後回しにする" == child_action_name[a]:
            locx = x-1
            locy = y-1
        if "掃除する" == child_action_name[a]:
            locx = x+1
            locy = y
        for next_parent_emotion in emo_list:
            next_x = next_parent_emotion[0]
            next_y = next_parent_emotion[1]
            B[0][emo_list.index((next_x, next_y)), emo_list.index((x, y)), a] = (stats.norm.pdf(x = next_x, loc=locx, scale=1.0) + stats.norm.pdf(x = next_y, loc=locy, scale=1.0))/2


# # p(s2|s2, a2)
# B[1][parent_action_name.index("praise"), parent_action_name.index("praise"), child_action_name.index("Do")] = 1.0
# B[1][parent_action_name.index("praise"), parent_action_name.index("praise"), child_action_name.index("Don't")] = 0.3
# B[1][parent_action_name.index("praise"), parent_action_name.index("nutral"), child_action_name.index("Do")] = 1.0
# B[1][parent_action_name.index("praise"), parent_action_name.index("nutral"), child_action_name.index("Don't")] = 0.2
# B[1][parent_action_name.index("praise"), parent_action_name.index("scold"), child_action_name.index("Do")] = 1.0
# B[1][parent_action_name.index("praise"), parent_action_name.index("scold"), child_action_name.index("Don't")] = 0

# B[1][parent_action_name.index("nutral"), parent_action_name.index("praise"), child_action_name.index("Do")] = 0
# B[1][parent_action_name.index("nutral"), parent_action_name.index("praise"), child_action_name.index("Don't")] = 0.6
# B[1][parent_action_name.index("nutral"), parent_action_name.index("nutral"), child_action_name.index("Do")] = 0
# B[1][parent_action_name.index("nutral"), parent_action_name.index("nutral"), child_action_name.index("Don't")] = 0.5
# B[1][parent_action_name.index("nutral"), parent_action_name.index("scold"), child_action_name.index("Do")] = 0
# B[1][parent_action_name.index("nutral"), parent_action_name.index("scold"), child_action_name.index("Don't")] = 0.2

# B[1][parent_action_name.index("scold"), parent_action_name.index("praise"), child_action_name.index("Do")] = 0
# B[1][parent_action_name.index("scold"), parent_action_name.index("praise"), child_action_name.index("Don't")] = 0.1
# B[1][parent_action_name.index("scold"), parent_action_name.index("nutral"), child_action_name.index("Do")] = 0
# B[1][parent_action_name.index("scold"), parent_action_name.index("nutral"), child_action_name.index("Don't")] = 0.3
# B[1][parent_action_name.index("scold"), parent_action_name.index("scold"), child_action_name.index("Do")] = 0
# B[1][parent_action_name.index("scold"), parent_action_name.index("scold"), child_action_name.index("Don't")] = 0.8


B = utils.norm_dist_obj_arr(B)

# p(~o)
C = utils.obj_array_zeros(num_obs)

# p(~o1) = 0
for parent_emotion in emo_list:
    if -x_limit <= parent_emotion[0] < -x_limit/3:
        C[0][emo_list.index(parent_emotion)] = -5
    if -x_limit/3 <= parent_emotion[0] <= x_limit/3:
        C[0][emo_list.index(parent_emotion)] = 0
    if x_limit/3 < parent_emotion[0] <= x_limit:
        C[0][emo_list.index(parent_emotion)] = 0

    if -y_limit <= parent_emotion[1] < -y_limit/3:
        C[0][emo_list.index(parent_emotion)] = -2.5
    if -y_limit/3 <= parent_emotion[1] <= y_limit/3:
        C[0][emo_list.index(parent_emotion)] = 0
    if y_limit/3 < parent_emotion[1] <= y_limit:
        C[0][emo_list.index(parent_emotion)] = 0

for child_emotion in emo_list:
    if -x_limit <= child_emotion[0] < -x_limit/3:
        C[1][emo_list.index(child_emotion)] = 0
    if -x_limit/3 <= child_emotion[0] <= x_limit/3:
        C[1][emo_list.index(child_emotion)] = 0
    if x_limit/3 < child_emotion[0] <= x_limit:
        C[1][emo_list.index(child_emotion)] = 3

    if -y_limit <= child_emotion[1] < -y_limit/3:
        C[1][emo_list.index(child_emotion)] = 0
    if -y_limit/3 <= child_emotion[1] <= y_limit/3:
        C[1][emo_list.index(child_emotion)] = 0
    if y_limit/3 < child_emotion[1] <= y_limit:
        C[1][emo_list.index(child_emotion)] = 1.5

# p(~o2)
# for child_emotion in emo_list:
#     if -x_limit <= child_emotion[0] < -x_limit/3:
#         C[1][emo_list.index(child_emotion)] = -3
#     if -x_limit/3 <= child_emotion[0] <= x_limit/3:
#         C[1][emo_list.index(child_emotion)] = 0
#     if x_limit/3 < child_emotion[0] <= x_limit:
#         C[1][emo_list.index(child_emotion)] = 400

# p(~o3)
# C[2][parent_action_name.index("praise")] = 0
# C[2][parent_action_name.index("nutral")] = 0
# C[2][parent_action_name.index("scold")] = 40


# p(s)
D = utils.obj_array_zeros(num_states)
# D = utils.obj_array_ones(num_states)
# D = utils.norm_dist_obj_arr(D)



# Agent
child = Agent_variable()
parent = Agent_variable()



# def parent_act_fun(x, a, init = False):
#     # speak
#     speak = ""
#     m = x_limit * y_limit
#     for e in index_emo:
#         distance = np.linalg.norm([speak_emo[e][0]-x[0], speak_emo[e][1]-x[1]], ord=2)
#         if distance < m:
#             m = distance
#             speak = e
#         else:
#             continue

#     if init == True:
#         return [speak, a]

#     # act
#     act = ""
#     if a == child_action_name[0]:
#         act = parent_action_name[0]
#     elif (x[0] < -x_limit/2) and (y_limit/2 < x[1]):
#         act = parent_action_name[2]
#     else:
#         act = parent_action_name[1]

#     return [speak, act]


def fun_child_x(x, y):
    if y >= x:
        f = - pow(0.7, y-x) + 1
        return f
    else:
        f = pow(0,7, x-y) - 1
        return f


def child_x_update(x, y, a):
    gauss = stats.norm.rvs(loc=-(child_action_name.index(a)-2)/2, scale=1.5, size=1)[0]
    x[0] = min(x_limit, max(-x_limit, x[0] + round(fun_child_x(x[0], y[0]) + gauss)))
    x[1] = min(y_limit, max(-y_limit, x[1] + round(fun_child_x(x[1], y[1]) + gauss)))
    return x



def fun_parent_x(x, y):
    if y >= x:
        gauss = stats.norm.rvs(loc=1, scale=1, size=1)
        f = - pow(0.7, y-x) + 1
        return x + round(gauss[0] + f)
    else:
        gauss = stats.norm.rvs(loc=-1, scale=1, size=1)
        f = pow(0,7, x-y) - 1
        return x + round(gauss[0] + f)

def parent_x_update(x, y):
    x[0] = min(x_limit, max(-x_limit, fun_parent_x(x[0], y[0])))
    x[1] = min(y_limit, max(-y_limit, fun_parent_x(x[1], y[1])))

    return x


def qs_graph(data, t, title):
    if title == "state":
        grid = np.empty((grid_dims[0], grid_dims[1]))
        for point in emo_list:
            value = data[emo_list.index(point)]
            grid[point[1]+10][point[0]+10] = value.copy()
        fig_qs_state = plt.figure()
        x_grid = [-10, "", "", "", "", -5, "", "", "", "", 0, "", "", "", "", 5, "", "", "", "", 10]
        y_grid = [10, "", "", "", "", 5, "", "", "", "", 0, "", "", "", "", -5, "", "", "", "", -10]
        fig_qs_state = sns.heatmap(data=grid, square=True, cmap='cool', vmax=1.0, vmin=0, xticklabels=x_grid, yticklabels=y_grid)
        fig_qs_state.set_title(f"qs2 parent state  {t}times")
        fig_qs_state = plt.savefig(Rf".\qs2_graph\qs_state_{t}times.png")
        fig_qs_state = plt.close()

    if title == "pstate":
        grid = np.empty((grid_dims[0], grid_dims[1]))
        for point in emo_list:
            value = data[emo_list.index(point)]
            grid[point[1]+10][point[0]+10] = value.copy()
        fig_qs_state = plt.figure()
        x_grid = [-10, "", "", "", "", -5, "", "", "", "", 0, "", "", "", "", 5, "", "", "", "", 10]
        y_grid = [10, "", "", "", "", 5, "", "", "", "", 0, "", "", "", "", -5, "", "", "", "", -10]
        fig_qs_state = sns.heatmap(data=grid, square=True, cmap='cool', vmax=1.0, vmin=0, xticklabels=x_grid, yticklabels=y_grid)
        fig_qs_state.set_title(f"ps2 parent state  {t}times")
        fig_qs_state = plt.savefig(Rf".\ps2_graph\ps_state_{t}times.png")
        fig_qs_state = plt.close()

    if title == "log-state":
        log_grid = np.empty((grid_dims[0], grid_dims[1]))
        for point in emo_list:
            value = data[emo_list.index(point)]
            log_grid[point[1]+10][point[0]+10] = round(np.log10(value), 1)
        fig_logqs = plt.figure()
        x_grid = [-10, "", "", "", "", -5, "", "", "", "", 0, "", "", "", "", 5, "", "", "", "", 10]
        y_grid = [10, "", "", "", "", 5, "", "", "", "", 0, "", "", "", "", -5, "", "", "", "", -10]
        fig_logqs = sns.heatmap(data=log_grid, square=True, cmap='cool', vmax=0, vmin=-20, xticklabels=x_grid, yticklabels=y_grid)
        fig_logqs.set_title(f"log(qs)2 parent state  {t}times")
        fig_logqs = plt.savefig(Rf".\qs2_graph\log(qs)_state_{t}times.png")
        fig_logqs = plt.close()

    if title == "action":
        act_grid = np.empty((1, num_states[1]))
        for i in range(num_states[1]):
            act_grid[0][i] = data[i].copy()
        fig_act_state = plt.figure()
        fig_act_state = sns.heatmap(data=act_grid, square=True, cmap='cool', vmax=1.0, vmin=0, xticklabels=['praise', 'nutral', 'scold'], yticklabels=False)
        fig_act_state.set_title(f"qs2 parent action  {t}times")
        fig_act_state = plt.savefig(Rf".\qs2_graph\qs_action_{t}times.png")
        fig_act_state = plt.close()





# Active Inference
def main(T, initial_parant_action, initial_parent_state, initial_child_state):
    D[0][emo_list.index((initial_parent_state[0], initial_parent_state[1]))] = 1
    # D[1][initial_parant_action] = 1

    # Agent class
    my_agent = Agent(A=A, B=B, C=C, D=D, policy_len=2)

    # History initialization
    parent.history_obs.clear()
    parent.history_state.clear()
    child.history_obs.clear()
    child.history_state.clear()

    qs_x = []
    qs_logx = []
    qs_action = []
    G_list = []
    qpi_list = []
    # Initial variable(parent)
    parent.x = initial_parent_state
    parent.a = initial_parant_action
    child.x = (initial_child_state)

    with open(R".\qs2_parent_state.csv", 'w') as f:
        pass
    # with open(R".\qs_parent_action.csv", 'w') as g:
    #     pass
    with open(R".\log(qs)2_parent_state.csv", 'w') as h:
        pass

    shutil.rmtree(R".\qs2_graph")
    os.mkdir(R".\qs2_graph")
    parent.text = "部屋の掃除お願いね"
    print(f"Parent talks to child {parent.text}")

    for t in range(T):
        print(f"{t+1} times")

        # Parent text -> Child observation
        print(f"Parent hidden state {parent.x}")
        # print(f"Parent action {parent.a}")

        child.y, child.log_obs = child_obs(child.log_obs, parent.text)

        child.history_state.append(child.x.copy())
        child.history_obs.append(child.y.copy())
        parent.history_state.append(parent.x.copy())

        # Derivation of q(s) from observation
        qs = my_agent.infer_states([emo_list.index(tuple(child.y)), emo_list.index(tuple(child.x))])
        # qs report to csv
        with open(R".\qs2_parent_state.csv", 'a', newline='', encoding="utf8") as f:
            writer = csv.writer(f)
            writer.writerow(qs[0])

        with open(R".\log(qs)2_parent_state.csv", 'a', newline='', encoding="utf8") as h:
            writer = csv.writer(h)
            writer.writerow(np.round(np.log10(qs[0]), decimals=1))

        # with open(R".\qs_parent_action.csv", 'a', newline='', encoding="utf8") as g:
        #     writer = csv.writer(g)
        #     writer.writerow(qs[1])

        qs_graph(qs[0], t+1, "state")
        qs_graph(qs[0], t+1, "log-state")
        # qs_graph(qs[1], t+1, "action")

        qs_x.append(qs[0].copy())
        qs_logx.append(np.log10(qs[0]).copy())
        # qs_action.append(qs[1].copy())

        # Policy
        qpi = 0
        G = 0
        qpi, G = my_agent.infer_policies()
        qpi_list.append(qpi.copy())
        G_list.append(-G.copy())
        print(t, min(-G))

        # Devide action
        chosen_action_id = my_agent.sample_action()
        child.a = child_action_name[int(chosen_action_id[0])]
        child.x = child_x_update(child.x, child.y, child.a)
        print(f"Child action {child.a}")
        print(f"Child hidden state {child.x}")
        child.text, child.log_act = child_act(child.x, child.a, parent.text, child.log_act)
        parent.y, parent.log_obs = parent_obs(parent.log_obs, child.text)
        parent.history_obs.append(parent.y.copy())

        # Parent function
        # parent.a = parent_act_fun(parent.x, child.a[1])
        # parent.x = parent_x_update(parent.x, parent.y)

        parent.text, parent.log_act = parent_act(parent.x, child.text, parent.log_act)



        print("---------------------------------------------------------------")

    print("END")

    np.save("./data2/qs_x", qs_x)
    np.save("./data2/qs_logx", qs_logx)
    np.save("./data2/qs_action", qs_action)
    np.save("./data2/qpi", qpi_list)
    np.save("./data2/G", G_list)

    return (
        parent.history_obs,
        parent.history_state,
        child.history_obs,
        child.history_state,
    )
