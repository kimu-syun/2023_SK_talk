# p(o|s)を求める
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
import global_value as g

from gpt_child_act import chi_text_gpt
from gpt_child_obs import chi_obs_gpt
from gpt_parent_act import par_text_gpt

g.gpt_model = "gpt-3.5-turbo"
# g.gpt_model = "gpt-4"

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")


# sを指定
# s = [-5, -5]

# 回す回数
n = 100

# x = -5
for x in range(5, 6):
    for y in range(-2, 6):
        s = [x, y]
        cnt = np.zeros((11, 11))

        for i in range(0, n):
            # GPTに入れてs -> o
            par_text = par_text_gpt(par_emo=s, first=True)
            o = chi_obs_gpt(par_text)
            print(o)
            print(par_text)

            # カウント
            cnt[5-o[1], 5+o[0]] += 1
        print(cnt)
        np.savetxt(Rf".\A\csv\{s}.csv", cnt)

        # 分布で表示
        cnt = cnt / n
        A = plt.figure
        x_grid = [-5, "", "", "", "", 0, "", "", "", "", 5]
        y_grid = [5, "", "", "", "", 0, "", "", "", "", -5]
        A = sns.heatmap(data=cnt, square=True, cmap='cool', xticklabels=x_grid, yticklabels=y_grid, linewidths=0.5)
        A.set_title(f"p(o|{s})")
        A = plt.plot(s[0]+5.5, 5.5-s[1], marker=".", color="black", markersize=15)
        A = plt.savefig(Rf".\A\png\{s}.png")
        A = plt.close()

        np.save(Rf".\A\dis\{s}", cnt)
        print(np.load(Rf"A\dis\{s}.npy"))
