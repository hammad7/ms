# -*- coding: utf-8 -*-
"""DQN_Agent_Arch2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11hZOprjUvb0RHBxWL4A4Fxxonz1Gv9H0

### Cab-Driver Agent
"""

# Importing libraries
import numpy as np
import random
import math
from collections import deque
import collections
import pickle

# for building DQN model
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

# for plotting graphs
import matplotlib.pyplot as plt

# Import the environment
from Env import CabDriver

"""#### Defining Time Matrix"""

# Loading the time matrix provided
Time_matrix = np.load("TM.npy")
Time_matrix.shape

"""#### Tracking the state-action pairs for checking convergence

"""







#Defining a function to save the Q-dictionary as a pickle file
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

"""### Agent Class

If you are using this framework, you need to fill the following to complete the following code block:
1. State and Action Size
2. Hyperparameters
3. Create a neural-network model in function 'build_model()'
4. Define epsilon-greedy strategy in function 'get_action()'
5. Complete the function 'append_sample()'. This function appends the recent experience tuple <state, action, reward, new-state> to the memory
6. Complete the 'train_model()' function with following logic:
   - If the memory size is greater than mini-batch size, you randomly sample experiences from memory as per the mini-batch size and do the following:
      - Initialise your input and output batch for training the model
      - Calculate the target Q value for each sample: reward + gamma*max(Q(s'a,))
      - Get Q(s', a) values from the last trained model
      - Update the input batch as your encoded state-action and output batch as your Q-values
      - Then fit your DQN model using the updated input and output batch.
"""
R_TRACK = 5
class DQNAgent:
    def __init__(self, state_size, action_size):
        # Define size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # Write here: Specify you hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.max_epsilon = 1
        self.epsilon = self.max_epsilon
        self.decay_rate = 0.001
        self.min_epsilon = 0.01
        self.batch_size = 32        
        # create replay memory using deque
        self.memory = deque(maxlen=2000)
        self.avg_rew = deque(maxlen = R_TRACK)

        # create main model and target model
        self.model = self.build_model()

        # Initialize the value of the states tracked for all samples
        self.states_tracked_1 = []
        self.states_tracked_2 = []
        # For Sample state-action pair 1: We are going to track state (2,4,6) and action (2,3) at index 11 in the action space.
        self.track_state_1 = np.array(env.state_encod_arch2((2,4,6))).reshape(1, self.state_size)
        # For Sample state-action pair 2: We are going to track state (1,2,3) and action (1,2) at index 6 in the action space.
        self.track_state_2 = np.array(env.state_encod_arch2((1,2,3))).reshape(1, self.state_size)


    # approximate Q function using Neural Network
    def build_model(self):
        model = Sequential()
        # Write your code here: Add layers to your neural nets       
        model.add(Dense(32, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        # model.add(Dense(32, activation='relu',
        #                 kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model



    def get_action(self, state):
        # Write your code here:
        # get action from model using epsilon-greedy policy
        # Decay in ε after we generate each sample from the environment       
        
        # self.epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*time)

        z = np.random.random()
        if z <= self.epsilon:
            return random.choice(env.action_space)
        else:
            # choose the action with the highest q(s, a)
            # the first index corresponds to the batch size, so
            # reshape state to (1, state_size) 
            state = np.array(env.state_encod_arch2(state)).reshape(1, self.state_size)

            # Use the model to predict the Q_values.
            action = self.model.predict(state)
            # return action
            # q_vals_possible = [action[0][i] for i in env.action_space]

            return env.action_space[np.argmax(action[0])]
    

    def append_sample(self, state, action, reward, next_state,done):
        # Write your code here:
        # save sample <s,a,r,s'> to the replay memory
        self.memory.append((state, action, reward, next_state, done))
    

    # pick samples randomly from replay memory (with batch_size) and train the network
    def train_model(self):
        if len(self.memory) > self.batch_size:
            # Sample batch from the memory
            mini_batch = random.sample(self.memory, self.batch_size)
            update_output = np.zeros((self.batch_size, self.state_size))
            update_input = np.zeros((self.batch_size, self.state_size))
            actions, rewards, dones = [], [], []
            
            for i in range(self.batch_size):
                
                # Write your code from here
                state, action, reward, next_state, done = mini_batch[i]
                update_input[i] = env.state_encod_arch2(state)     
                actions.append(action)
                rewards.append(reward)
                update_output[i] = env.state_encod_arch2(next_state)
                dones.append(done)
            
            # 1. Predict the target from earlier model, Optimization preventing calls to multiple predicts
            # target = self.model.predict(update_input, max_queue_size=100, workers=2, use_multiprocessing=True)
            # target_qval = self.model.predict(update_output, max_queue_size=100, workers=2, use_multiprocessing=True)
            target = self.model.predict(np.vstack([update_input,update_output]))
            target_qval = target[int(len(target)/2):int(len(target))]
            target = target[0:int(len(target)/2)]
            # 2. Get the target for the Q-network
            
            
            #3. Update your 'update_output' and 'update_input' batch. Be careful to use the encoded state-action pair
            for i in range(self.batch_size):
              if dones[i]:
                  target[i][env.action_encod(actions[i])] = rewards[i]
              else: # non-terminal state
                  # print(actions[i])
                  # print(target[i][actions[i]])
                  target[i][env.action_encod(actions[i])] = rewards[i] + self.discount_factor * np.max(target_qval[i])
                
            # 4. Fit your model and track the loss values        
            self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)
                
                
        
    def save_tracking_states(self):
        # Use the model to predict the q_value of the state we are tacking.
        # q_value_1 = self.model.predict(self.track_state_1)
        # q_value_2 = self.model.predict(self.track_state_2)
        q_value_1 = self.model.predict(np.vstack([self.track_state_1,self.track_state_2]))
        q_value_2 = q_value_1[int(len(q_value_1)/2):int(len(q_value_1))]
        q_value_1 = q_value_1[0:int(len(q_value_1)/2)]

        
        # Grab the q_value of the action index that we are tracking.
        self.states_tracked_1.append(q_value_1[0][11])    ## action (2,3) at index 11 in the action space
        self.states_tracked_2.append(q_value_2[0][6])     ## action (1,2) at index 6 in the action space


    def save(self, name):
        self.model.save_weights(name)

episode_time = 24*30   # 24 hrs for 30 days per episode
Episodes = 5000       # No. of Episodes

m = 5                  # No. Locations
t = 24                 # No. of hrs in a day
d = 7                  # No. of days in a week 
state_size = m+t+d  ## require as len for the encoded  nn input

# Invoke Env class
env = CabDriver()
# state = env.reset()

# Invoke agent class
agent = DQNAgent(action_size=len(env.action_space), state_size = state_size)  ### len(env.state_space)

# to store rewards in each episode
rewards_per_episode, episodes = [], []
# Rewards for state
rewards_init_state = []
score_tracked = []

"""### DQN block"""

for episode in tqdm(range(Episodes)):

    terminal_state = False
    score = 0

    # Reset at the start of each episode
    state = env.reset()
    # State Initialization 
    initial_state = state


    total_time = 0  
    while not terminal_state:    
        # 1. Pick epsilon-greedy action from possible actions for the current state
        action = agent.get_action(state)
        # 2. Evaluate your reward and next state
        next_state, reward, terminal_state = env.step(state, action, Time_matrix)
        # 3. Append the experience to the memory
        agent.append_sample(state, action, reward, next_state, terminal_state)
        # 4. Train the model by calling function agent.train_model
        agent.train_model()
        # 5. Keep a track of rewards, Q-values, loss
        score += reward
        state = next_state
        total_time+=1
        
    # Store total reward obtained in this episode
    rewards_per_episode.append(score)
    episodes.append(episode)
    agent.avg_rew.append(score)
        
    agent.epsilon = agent.min_epsilon + (agent.max_epsilon - agent.min_epsilon) * np.exp(-agent.decay_rate*episode)

    # Every 10 episodes:
    if ((episode + 1) % R_TRACK == 0):
        print("episode {0}, initial_state {1}, avg_reward {2}, memory_length {3}, epsilon {4} total_epochs {5} episode_time {6}".format(episode+1, 
                                                                         initial_state,
                                                                         np.mean(agent.avg_rew),
                                                                         len(agent.memory),
                                                                         agent.epsilon, total_time, env.episode_time))
    # Total rewards per episode
    score_tracked.append(score)

    # Save the Q_value of the state-action pair we are tracking (every 5 episodes)
    if ((episode + 1) % 5 == 0):
        agent.save_tracking_states()


    ## Saving the 'DQN_model' and 'model_weights' every 1000th episode.
    if(episode % 1000 == 0):
        save_obj(score_tracked, "score_tracked")
        print("Saving Model {}".format(episode))
        agent.save(name="DQN_model.h5")                     ## Saves DQN model in Keras H5 format
        save_obj(agent.states_tracked_1,"states_tracked_1")
        save_obj(agent.states_tracked_2,"states_tracked_2")
        # print("Saving Model {} Weights".format(episode))
        # agent.save_weights_numpy(name="model_weights.pkl")





"""### Tracking Convergence"""

def load_obj(name ):
    obj = None
    with open(name, 'rb') as f:
        obj = pickle.load(f)
    return obj


rewards = load_obj("score_tracked.pkl")
states_tracked_1 = load_obj("states_tracked_1.pkl")
states_tracked_2 = load_obj("states_tracked_2.pkl")


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





"""#### Epsilon-decay sample function

<div class="alert alert-block alert-info">
Try building a similar epsilon-decay function for your model.
</div>
"""

time = np.arange(0,Episodes)
epsilon = []
for i in range(0,Episodes):
    epsilon.append(0 + (1 - 0) * np.exp(-agent.decay_rate*i))
  
plt.plot(time, epsilon)
plt.show()

