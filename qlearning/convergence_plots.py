from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product
from TC_Env import TicTacToe
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


rewards = load_obj("Rewards.pkl")
deltas = load_obj("Deltas.pkl")


plt.plot(deltas[0::100], label="deltas")
plt.legend()
plt.title("Convergence Plot - Q-Learning")
plt.tight_layout()
plt.show()

plt.plot( rewards[0::100], label="rewards")
plt.legend()
plt.title("Convergence Plot - Q-Learning")
plt.tight_layout()
plt.show()