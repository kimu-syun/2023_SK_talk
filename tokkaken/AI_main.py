# 能動的推論を使った会話モデル
import numpy as np
from math import log
import matplotlib.pyplot as plt
import C_talk, P_talk

# agentの情報保存
class agent:

    def __init__(self):
        # x,y,a
        self.x = 0
        self.y = 0
        self.a = 0
        self.u = 0
        # FE求めるのに必要なやつ
        self.epistemic_value = np.zeros((5, 100))
        self.predicted_surprised = np.zeros((5, 100))
        self.F_expected = np.zeros((5, 100))
        # 分布の設定
        self.Pyxa = np.zeros((y_range, x_range, a_range)) # p(y|x,a)
        self.Pstack = np.ones((y_range, x_range, a_range))
        self.Qxa = np.zeros((x_range, a_range, 100)) # q(x|a)
        self.Py = np.zeros((5, y_range)) # p~(y)
        self.Qya = np.zeros((y_range, a_range, 100)) # q(y|a)
        self.Qxya = np.zeros((x_range, y_range, a_range, 100)) # q(x|y,a)

        # 記録
        self.F_save = np.zeros((5, epoch))
        self.PS_save = np.zeros((5, epoch))
        self.EV_save = np.zeros((5, epoch))
        self.Fmin_save = np.zeros(epoch)
        self.a_save = np.zeros(epoch)
        self.y_save = np.zeros(epoch)



# x,y,aの範囲
emotion_range = 5
relation_range = 5
y_range = 5
a_range = 5
x_range = 5
hidden_state_range = emotion_range * relation_range

# active inference
epoch = 100
a_order = [0, 0, 0, 0, 0]
y_sum = [0, 0, 0, 0, 0]
a_sum = [0, 0, 0, 0, 0]


# 真の値
child = agent()
# parent = agent()


# 子供の分布の作成(p(y|x,a), q(x|a), p~(y))
child.Pyxa = C_talk.Pyxa_make(child)
child.Qxa = C_talk.Qxa_make(child.Qxa)
child.Py = C_talk.Py_make(child.Py)


# 親の関数
parent = P_talk.agent_function()
P_x = 2
P_a = 0
P_aini = 2



# parent.action = 1 #親の初期行動
P_a = P_aini

for i in range(0, epoch):
    print(f"{i+1}回目")

    # 親の行動⇒子の感覚
    child.y = P_a

    # 子の推論
    print(f"子供の感覚信号 : {child.y}")
    child = C_talk.child_inference(child, i)


    # ふるまい記録
    y_sum[child.y] += 1
    if child.y == 0:
        if child.a == 2 or child.a == 3 or child.a == 4:
            a_sum[0] += 1
    if child.y == 1:
        if child.a == 1 or child.a == 2 or child.a == 3 or child.a == 4:
            a_sum[1] += 1
    if child.y == 2:
        if child.a == 0 or child.a == 1 or child.a == 2 or child.a == 3 or child.a == 4:
            a_sum[2] += 1
    if child.y == 3:
        if child.a == 0 or child.a == 1 or child.a == 2 or child.a == 3:
            a_sum[3] += 1
    if child.y == 4:
        if child.a == 0 or child.a == 1 or child.a == 2:
            a_sum[4] += 1



    print(f"子供の行動{child.a}")

    P_y = child.a
    # 親の関数
    P_x, P_a = parent.fun(P_x, P_y)
    print(f"親 x={P_x}, a={P_a}")
    print()




# F出力
fig = plt.figure(figsize=(15, 10))
fig.suptitle(f"Active Inference - parent initial action = {P_aini}")

ax1 = fig.add_subplot(2, 1, 1)

# ax1.set_xlabel("time steps")
# ax1.set_ylabel("期待自由エネルギーF")
# ax1.grid()

# ax1.plot(child.F_save[0], label = "action0")
# ax1.plot(child.F_save[1], label = "action1")
# ax1.plot(child.F_save[2], label = "action2")
# ax1.plot(child.F_save[3], label = "action3")
# ax1.plot(child.F_save[4], label = "action4")

ax1.set_xlabel("time step")
ax1.set_ylabel("a,yの遷移")
ax1.grid()
ax1.plot(child.y_save, label="y")
ax1.plot(child.a_save, label="a")




# ax2 = fig.add_subplot(2, 2, 2)
# ax2.set_xlabel("time steps")
# ax2.set_ylabel("epistimec value")
# ax2.grid()
# ax2.plot(child.EV_save[0], label = "action0")
# ax2.plot(child.EV_save[1], label = "action1")
# ax2.plot(child.EV_save[2], label = "action2")
# ax2.plot(child.EV_save[3], label = "action3")
# ax2.plot(child.EV_save[4], label = "action4")


# ax3 = fig.add_subplot(2, 2, 3)
# ax3.set_xlabel("time steps")
# ax3.set_ylabel("predicted surprised")
# ax3.grid()
# ax3.plot(child.PS_save[0], label = "action0")
# ax3.plot(child.PS_save[1], label = "action1")
# ax3.plot(child.PS_save[2], label = "action2")
# ax3.plot(child.PS_save[3], label = "action3")
# ax3.plot(child.PS_save[4], label = "action4")


ax4 = fig.add_subplot(2, 1, 2)
ax4.set_xlabel("time steps")
ax4.set_ylabel("minF")
ax4.grid()
ax4.plot(child.Fmin_save, label = "FE")


ax1.legend()
# ax2.legend()
# ax3.legend()
ax4.legend()
# ax1.legend(loc="lower left", bbox_to_anchor=(1.02, 0.0,), borderaxespad=0)
# ax2.legend(loc="lower left", bbox_to_anchor=(1.02, 0.0,), borderaxespad=0)
# ax3.legend(loc="lower left", bbox_to_anchor=(1.02, 0.0,), borderaxespad=0)
# plt.legend(["action0", "action1", "action2", "action3", "action4"])

Y = np.array([0, 0, 0, 0, 0])
for i in range(0, 5):
    Y[i] = (y_sum[i] / epoch) * 100
print("yの出現確率")
print(Y)
#ふるまい出力
for i in range(0, 5):
    if y_sum[i] == 0:
        continue
    a_order[i] = a_sum[i] * 100 / (y_sum[i])

print("正しく返せたaの割合")
print(a_order)
print(y_sum)

plt.show()