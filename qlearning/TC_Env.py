from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product

###
# DEfining MDP
#
# THIS CLASS IS KIND OF STATIC CLASS UTILITIES, elf.sta never modified
# 
# The environment is playing randomly with the agent, i.e. its strategy is to put an even number randomly in an empty cell. 
# If the agent wins the game, it gets 10 points, if the environment wins, the agent loses 10 points. 
# And if the game ends in a draw, it gets 0. Also, the agent needs to win in as few moves as possible, so for each move, it gets a -1 point.
###
class TicTacToe():

    def __init__(self):
        """initialise the board"""
        
        # initialise state as an array
        # self.state = [np.nan for _ in range(9)]  
        self.state = self.reset()  # initialises the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array or matrix

        self.reset()


    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
        return curr_state[0]+curr_state[1]+curr_state[2]==15 or curr_state[3]+curr_state[4]+curr_state[5]==15 or \
        curr_state[6]+curr_state[7]+curr_state[8]==15 or \
        curr_state[0]+curr_state[3]+curr_state[6]==15 or curr_state[1]+curr_state[4]+curr_state[7]==15 or \
        curr_state[2]+curr_state[5]+curr_state[8]==15 or curr_state[0]+curr_state[4]+curr_state[8]==15 or \
        curr_state[2]+curr_state[4]+curr_state[6]==15


    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) ==0:
            return True, 'Tie'

        else:
            return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)


    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)



    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        # assert
        curr_state[curr_action[0]] = curr_action [1]
        return curr_state


    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
        final_state, reward = None, None

        agent_action_state = self.state_transition(curr_state, curr_action) ## incorporate user action
        isTerminal, game_state = self.is_terminal(agent_action_state)
        if(not isTerminal):
            ####### env takes random action, env not clever/intellectual, but agent learns
            pos = random.choice(self.allowed_positions(agent_action_state))
            val = random.choice(self.allowed_values(agent_action_state)[1])
            curr_action = (pos,val)

            env_action_state = self.state_transition(agent_action_state, curr_action) ## incorporate user action
            isTerminal, game_state = self.is_terminal(env_action_state)
            final_state = env_action_state
            if(not isTerminal):  ### agent move, enforces agent to quickly learn to win
                reward = -1
            else:
                if game_state == "Win":
                    reward = -10
                else:
                    reward = 0
        else:
            final_state = agent_action_state
            if game_state == "Win":
                reward = 10
            else:
                reward = 0
        return (final_state, reward, isTerminal)


    def reset(self):
        self.state = [np.nan for _ in range(9)]
        return self.state
