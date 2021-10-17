import numpy as np
import random
from itertools import groupby
from itertools import product
import collections
import pickle
import time
from tqdm import tqdm
from matplotlib import pyplot as plt


def load_obj(name ):
    obj = None
    with open(name, 'rb') as f:
        obj = pickle.load(f)
    return obj


rewards = load_obj("score_tracked.pkl")
states_tracked_1 = load_obj("states_tracked_1.pkl")
states_tracked_2 = load_obj("states_tracked_2.pkl")

# print(len(rewards))

# plt.plot(deltas[0::100], label="deltas")
# plt.legend()
# plt.title("Convergence Plot - Q-Learning")
# plt.tight_layout()
# plt.show()

plt.plot( rewards, label="rewards")
plt.legend()
plt.title("Convergence Plot - DQN")
plt.tight_layout()
plt.show()


def convergence_graph_q_val(fig_num, state, action, states_tracked):
    plt.figure(fig_num, figsize=(10,4))
    plt.title(f"Convergence of Q_values for state {state} and action {action}", fontsize=14, fontweight='bold')
    xaxis = np.asarray(range(0, len(states_tracked)))
    plt.plot(xaxis,np.asarray(states_tracked))
    plt.ylabel("Q_values", fontsize=13, fontstyle='italic')
    plt.xlabel("No. of Episodes  (step increment of 5)", fontsize=13, fontstyle='italic')
    plt.show()

print('\n\033[1m'+"Tracking Convergence for state-action pair 1: State (2,4,6), Action (2,3)\n")
convergence_graph_q_val(fig_num=1, state=(2,4,6), action=(2,3), states_tracked= states_tracked_1)

print('\n\n\n\033[1m'+"Tracking Convergence for state-action pair 2: State (1,2,3), Action (1,2)\n")
convergence_graph_q_val(fig_num=3, state=(1,2,3), action=(1,2), states_tracked= states_tracked_2)