# 子供の能動推論(P_talkに対応)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from math import log
import random
import pandas as pd



# x,y,aの範囲
emotion_range = 5
relation_range = 5
sensory_range = 5
action_range = 5
hidden_state_range = emotion_range * relation_range


# xの変換(p(y|x,a))
def trans_gx(x):
    out = 0
    if x == 0:
        out = -0.6
    elif x == 1:
        out = -0.3
    elif x == 2:
        out = 0
    elif x == 3:
        out = 0.3
    elif x == 4:
        out = 0.6
    return out



# # 信念分布q(x|a)
# def Qxa_make(Qxa):
#     for a in range(0, 5):
#         sum = 0
#         for x in range(0, 5):
#             Qxa[x, a] = round(stats.norm.pdf(x=x, loc=a, scale=0.8), 4)
#             sum += Qxa[x, a]
#         # 正規化
#         for x in range(0, 5):
#             Qxa[x, a] = round(Qxa[x, a] / float(sum), 4)
#             if Qxa[x, a] == 0:
#                 Qxa[x, a] = 0.001
#     return Qxa


# 信念分布q(x|a)
# 最初を全部2中心の正規分布にする
def Qxa_make(Qxa):
    e = np.linspace(0, 4, 100)
    for a in range(0, 5):
        for u in range(0, 100):
            sum = 0
            for x in range(0, 5):
                Qxa[x, a, u] = round(stats.norm.pdf(x=x, loc=e[u], scale=0.4), 4)
                sum += Qxa[x, a, u]
        # 正規化
            for x in range(0, 5):
                Qxa[x, a, u] = round(Qxa[x, a, u] / float(sum), 4)
                if Qxa[x, a, u] == 0:
                    Qxa[x, a, u] = 0.001
    return Qxa

# 尤度分布p(y|x,a)
# def Pyxa_make(Pyxa):
#     for y in range(0, 5):
#         sum = 0
#         for x in range(0, 5):
#             for a in range(0, 5):
#                 Pyxa[y, x, a] = round(stats.norm.pdf(x=y, loc=a+trans_gx(x), scale=1.0), 4)
#                 sum += Pyxa[y, x, a]
#         # 正規化
#         for x in range(0, 5):
#             for a in range(0, 5):
#                 Pyxa[y, x, a] = round(Pyxa[y, x, a] / float(sum), 4)
#                 if Pyxa[y, x, a] == 0:
#                     Pyxa[y, x, a] = 0.001
#     return Pyxa

def Pyxa_make(child):
    # 確率に変換
    for y in range(0, 5):
        for x in range(0, 5):
            for a in range(0, 5):
                child.Pyxa[y, x, a] = round(child.Pstack[y, x, a] / float(5), 4)
                if child.Pyxa[y, x, a] == 0:
                    child.Pyxa[y, x, a] = 0.001
    return child.Pyxa

# 選好分布(p~(y))
def Py_make(Py):
    Py[0] = np.array([0.001, 0.001, 1/3, 1/3, 1/3])
    Py[1] = np.array([0.001, 1/4, 1/4, 1/4, 1/4])
    Py[2] = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
    Py[3] = np.array([1/4, 1/4, 1/4, 1/4, 0.001])
    Py[4] = np.array([1/3, 1/3, 1/3, 0.001, 0.001])
    return Py


# def Py_make(Py):
#     Py[0] = np.array([0.001, 0.001, 1/15, 1/15, 1/15])
#     Py[1] = np.array([0.001, 1/20, 1/20, 1/20, 1/20])
#     Py[2] = np.array([1/25, 1/25, 1/25, 1/25, 1/25])
#     Py[3] = np.array([1/20, 1/20, 1/20, 1/20, 0.001])
#     Py[4] = np.array([1/15, 1/15, 1/15, 0.001, 0.001])
#     return Py

# q(y|a)
def Qya_make(Qya, Pyxa, Qxa):
    for y in range(0, 5):
        for a in range(0, 5):
            for x in range(0, 5):
                for u in range(0, 100):
                    Qya[y, a, u] += Pyxa[y, x, a] * Qxa[x, a, u] # q(y|a) = Σ_x{p(y|x)*q(x)}
                if Qya[y, a, u] == 0:
                    Qya[y, a, u] = 0.001
    return Qya


# q(x|y,a)
def Qxya_make(Qxya, Pyxa, Qxa, Qya):
    for x in range(0, 5):
        for y in range(0, 5):
            for a in range(0, 5):
                for u in range(0, 100):
                    Qxya[x, y, a, u] = round((Pyxa[y, x, a] * Qxa[x, a, u]) / float(Qya[y, a, u]), 4) # q(x|y,a) = {p(y|x,a)q(x|a)}/q(y|a)
                    if Qxya[x,  y, a, u] == 0:
                        Qxya[x, y, a, u] = 0.001
    return Qxya


# epistemic value(エピスティミック価値)信念の更新度合い
def epistemic_value_calculate(Qya, Qxya, Qxa, epistemic_value):
    for a in range(0, 5):
        for y in range(0, 5):
            for x in range(0, 5):
                for u in range(0, 100):
                    epistemic_value[a, u] = Qya[y, a, u] * Qxya[x, y, a, u] * log(Qxya[x, y, a, u] / float(Qxa[x, a, u])) # Σ_y{q(y|a)Σ_x{q(x|y,a)log(q(x|y,a)/q(x|a))}}
    return epistemic_value


# predicted surprised(予測サプライズ)
def predicted_surprised_calculate(Qya, Py, s, predicted_surprised):
    for a in range(0, 5):
        for y in range(0, 5):
            for u in range(0, 100):
                predicted_surprised[a, u] -= Qya[y, a, u] * log(Py[s, y])
    return predicted_surprised


# 期待自由エネルギーを計算、保存
def F_expected_calculate(epistemic_value, predicted_surprised, F_expected):
    for a in range(0, 5):
        for u in range(0, 100):
            F_expected[a, u] = -epistemic_value[a, u] + predicted_surprised[a, u]
    return F_expected

# def UpdateQxa(Qxa, u, a):
#     sum = 0
#     e = np.linspace(0, 4, 100)
#     for x in range(0, 5):
#         Qxa[x, a, u] = round(stats.norm.pdf(x=x, loc=e[u], scale=0.4), 4)
#         sum += Qxa[x, a, u]
#     # 正規化
#     for x in range(0, 5):
#         Qxa[x, a, u] = round(Qxa[x, a] / float(sum), 4)
#         if Qxa[x, a, u] == 0:
#             Qxa[x, a, u] = 0.001
#     return Qxa


# minFE(a,u)求める、q更新
def Update(F_expected, Qxa):
    a, u = np.unravel_index(np.argmin(F_expected), F_expected.shape)
    # Qxa = UpdateQxa(Qxa, u, a)
    return a, Qxa, u


# q(x)を更新 q(x|y,a)->q(x)
def qxUpdate(Qx, Qxya, y, a):
    for x in range(0, 5):
        Qx[x, a] = Qxya[x, y, a]
    return Qx


def PyxaUpdate(child):
    #p(y|x,a)更新
    sum = 0
    for x in range(0, 5):
        child.Pstack[child.y, x, child.a] += child.Qxa[x, child.a, child.u]
    # 確率に変換
    for x in range(0, 5):
        for a in range(0, 5):
            sum = 0
            for y in range(0, 5):
                sum += child.Pstack[y, x, a]
            for y in range(0, 5):
                child.Pyxa[y, x, a] = round(child.Pstack[y, x, a] / float(sum), 4)

    return child.Pyxa


def child_inference(child, epoch):
    # FE,ev,ps初期化
    child.epistemic_value = np.zeros((5, 100))
    child.predicted_surprised = np.zeros((5, 100))
    child.F_expected = np.zeros((5, 100))
    child.Qya = np.zeros((5, 5, 100)) #q(y|a)
    child.Qxya = np.zeros((5, 5, 5, 100)) #q(x|y,a)
    child.u = 0

    # p(y|x,a)更新
    if not epoch == 0:
        child.Pyxa = PyxaUpdate(child)

    child.Qya = Qya_make(child.Qya, child.Pyxa, child.Qxa) #q(y|a)
    child.Qxya = Qxya_make(child.Qxya, child.Pyxa, child.Qxa, child.Qya) #q(x|y,a)

    # FE計算
    child.epistemic_value = epistemic_value_calculate(child.Qya, child.Qxya, child.Qxa, child.epistemic_value)
    child.predicted_surprised = predicted_surprised_calculate(child.Qya, child.Py, child.y, child.predicted_surprised)
    child.F_expected = F_expected_calculate(child.epistemic_value, child.predicted_surprised, child.F_expected)
    child.a, child.Qxa, child.u = Update(child.F_expected, child.Qxa)

    # 記録
    for j in range(0, 5):
        child.F_save[j, epoch] = child.F_expected[j, child.u]
        child.PS_save[j, epoch] = child.predicted_surprised[j, child.u]
        child.EV_save[j, epoch] = child.epistemic_value[j, child.u]
    child.Fmin_save[epoch] = child.F_expected[child.a, child.u]

    child.y_save[epoch] = child.y
    child.a_save[epoch] = child.a


    return child