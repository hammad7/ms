from TC_Env import TicTacToe
import collections
import numpy as np
import random
import pickle
import time
from tqdm import tqdm
from matplotlib import pyplot as plt


# # test
# rewards = 0
# for state in range(states):
#     best_action = np.argmax(q_table[state, :])
#     env.step
#     r = env.values[state][best_action]
#     rewards += r
# print("Reward collected: {}".format(rewards))



def load_obj(name ):
    obj = None
    with open(name, 'rb') as f:
        obj = pickle.load(f)
    return obj

Q_dict = load_obj("Policy.pkl")

def show_state(state):
    print("\n\n"+(' | '.join(str(e) for e in state[0:3])+' \n---------\n'+' | '.join(str(e) for e in state[3:6])+' \n---------\n'+' | '.join(str(e) for e in state[6:9])).replace('nan','x'))

# Function to convert state array into a string to store it as keys in the dictionary
# states in Q-dictionary will be of form: x-4-5-3-8-x-x-x-x
#   x | 4 | 5
#   ----------
#   3 | 8 | x
#   ----------
#   x | x | x
def Q_state(state):
    return ('-'.join(str(e) for e in state)).replace('nan','x')

is_terminal = False
env = TicTacToe()
while not is_terminal:
    curr_state = env.state
    curr_state_str = Q_state(env.state)
    agent_action = None
    if curr_state_str not in Q_dict:
        print("\nAgent not learnt/sampled current state, taking random action")
        pos = random.choice(env.allowed_positions(curr_state))
        val = random.choice(env.allowed_values(curr_state)[0])
        agent_action = (pos,val)
    else:
        agent_action = max(Q_dict[curr_state_str],key=Q_dict[curr_state_str].get)   ## decide action based on polic
    ## agent takes action
    curr_state = env.state_transition(curr_state, agent_action) 
    show_state(curr_state)
    is_terminal, game_state = env.is_terminal(curr_state)
    if is_terminal:
        print("Agent "+game_state)
        break
    ## take input
    pos_val = input("                                  Type space separated <pos[0-8] val[2,4,6,8]> :     ").split(" ")
    pos = int(pos_val[0])
    val = int(pos_val[1])
    ## incorporate user action
    curr_state = env.state_transition(curr_state, (pos,val)) 
    show_state(curr_state)
    is_terminal, game_state = env.is_terminal(curr_state)
    if is_terminal:
        print("You "+game_state)
        break


