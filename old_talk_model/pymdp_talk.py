# 特課研のコードをpymdpに書き換えたもの
import numpy as np
from pymdp.agent import Agent
from pymdp import utils, maths

# number of variable elements
num_obs = [5]
num_states = [5]
num_contorols = [5]

# general model(Child)
# p(o|s)
A_m_shapes = [[o_dim] + num_states for o_dim in num_obs]
A = utils.obj_array_zeros(A_m_shapes)
A[0][:, :] = np.eye(5)

# p(s|s,u)
B_f_shapes = [[ns, ns, num_contorols[f]] for f, ns in enumerate(num_states)]
B = utils.obj_array_zeros(B_f_shapes)

actions = ["So Bad", "Bad", "Neutral", "Good", "So Good"]
talks = ["So Angry", "Little Angry", "Neutral", "Little Prise", "Very Prise"]

for action_id, action in enumerate(actions):
    for last_state in range(num_states[0]):
        if action == "So Bad":
            if last_state > 1:
                next_state = last_state - 2
            else:
                next_state = 0
        elif action == "Bad":
            if last_state > 0:
                next_state = last_state - 1
            else:
                next_state = 0
        elif action == "Neutral":
            next_state = last_state
        elif action == "Good":
            if last_state < 4:
                next_state = last_state + 1
            else:
                next_state = 4
        elif action == "So Good":
            if last_state < 3:
                next_state = last_state + 2
            else:
                next_state = 4

        B[0][next_state, last_state, action_id] = 1.0

# p(~o)
C = utils.obj_array_zeros(num_obs)
C[0][4] = 2.0
C[0][0] = -2.0

# p(s)
D = utils.obj_array_uniform(num_states)

# general process
import parent_talk

parent = parent_talk.agent_function()

# Agent class
my_agent = Agent(A=A, B=B, C=C, D=D, policy_len=4)
parent_to_child = 1
parent_emotion = 2
obs = [parent_to_child]

# Active Inference
T = 10

for t in range(T):
    qs = my_agent.infer_states(obs)

    my_agent.infer_policies()
    chosen_action_id = my_agent.sample_action()

    talk_id = int(chosen_action_id[0])
    choice_action = actions[talk_id]

    print(f"Action at time {t}: child action {choice_action}")

    parent_emotion, parent_to_child = parent.fun(parent_emotion, talk_id)
    print(f"parent emotion {parent_emotion}")

    obs = [int(parent_to_child)]

    print(f"parent talk to child : {talks[parent_to_child]}")
    print('---------------------------------------------------------------')
