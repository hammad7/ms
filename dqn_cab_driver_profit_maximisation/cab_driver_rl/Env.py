# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""

        ## Retain (0,0) state and all states of the form (i,j) where i!=j
        self.action_space= [(p,q) for p in range(m) for q in range(m) if p!=q]
        self.action_space.insert(0, (0,0))         # no ride action   
        
        ## All possible combinations of (m,t,d) in state_space
        self.state_space = [(loc,time,day) for loc in range(m) for time in range(t) for day in range(d)]   
        
        ## Random state initialization
        self.state_init = self.reset()



    ## Encoding state (or state-action) for NN input

    def state_encod_arch2(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        curr_loc, curr_time, curr_day= state
        state_encod = [0 for _ in range(m+t+d)]

        state_encod[curr_loc] = 1
        state_encod[m+curr_time] = 1
        state_encod[m+t+curr_day] = 1

        return state_encod

    def action_encod(self, action):
        """convert the action into a vector so that it can be set as target of the NN. """
        return self.action_space.index(action)


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)


        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_idx]

        
        actions.append([0,0])

        return possible_actions_index,actions   



    def reward(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        return self.step(state, action, Time_matrix)[1]


    def next_state(self, state, action, Time_matrix):
        return self.step(state, action, Time_matrix)[0]

    def step(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state, reward"""
        total_time = None
        curr_loc, curr_time, curr_day = state
        p,q = action
        reward = 0
        done = False

        if action == (0,0): # reject request
            total_time = 1
            q = curr_loc
            reward = -C
        elif curr_loc!=p: # driver not present at pickpu location
            time_i_p = Time_matrix[curr_loc,p,curr_time,curr_day]
            curr_time_nxt, curr_day_nxt = self.update_time_by_duration(curr_time,curr_day, time_i_p)
            time_p_q = Time_matrix[p,q,curr_time_nxt,curr_day_nxt]
            total_time = time_i_p + time_p_q
            reward = R * time_p_q - C * (time_p_q + time_i_p)
        else:
            time_p_q = Time_matrix[p,q,curr_time,curr_day]
            total_time = time_p_q
            reward = R * time_p_q - C * (time_p_q)
        
        final_time, final_day = self.update_time_by_duration(curr_time,curr_day, total_time)
            
        next_state = q, final_time, final_day
        self.episode_time += total_time

        if self.episode_time > 720:
            done = True
        return next_state, reward, done



    def update_time_by_duration(self, curr_time, curr_day, ride_duration):
        """
        Add ride_duration to currebt timestamp to get updated timestamp
        """
        if (curr_time + ride_duration) < 24:
            updated_time = curr_time + ride_duration  
            updated_day= curr_day                       
        else:
            updated_time = (curr_time + ride_duration) % 24 
            num_days = (curr_time + ride_duration) // 24
            updated_day = (curr_day + num_days ) % 7

        return int(updated_time), int(updated_day)

    def reset(self):
        self.state_init = self.state_space[np.random.choice(len(self.state_space))]
        self.episode_time = 0  ## instance variable rest are class (as if static) variables
        return self.state_init


