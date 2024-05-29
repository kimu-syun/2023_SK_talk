import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as wg
import matplotlib.cm as cm
from pymdp.maths import softmax


qx = np.load("./data2/qs_x.npy")

parent_x = np.load("./data2/parent_state.npy")
init_data = np.load("./data2/init_data.npy", allow_pickle=True).item()
T = init_data["time"]
KL_list = []
dis_list = []

emo_list = []
for i in range(-10, 11):
    for j in range(-10, 11):
        emo_list.append((i, j))

for t in range(0, T):
    x = parent_x[t]
    px = np.full(441, 0.001)
    px[emo_list.index(tuple(x))] = 10
    px = softmax(px)
    kl = 0

    import gpt_talk
    gpt_talk.qs_graph(px, t+1, "pstate")

    maxq = 0
    maxq_index = np.array([0, 0])

    for i, q in enumerate(qx[t]):
        kl += q*np.log(q) - q*np.log(px[i])

        if maxq <= q:
            maxq = q
            maxq_index = np.array(list(emo_list[i]))
    KL_list.append(kl)

    dis = np.linalg.norm(maxq_index - np.array(x))
    dis_list.append(dis)

time = np.arange(1, T+1)

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax1.plot(time, KL_list)
ax2.plot(time, dis_list)

ax1.set_xlabel('time')
ax2.set_xlabel('time')

ax1.set_ylabel('KL divergence')
ax2.set_ylabel('2D distance')

ax1.set_xticks(time)
ax2.set_xticks(time)

ax1.grid()
ax2.grid()

plt.show()
print(KL_list)
print(dis_list)