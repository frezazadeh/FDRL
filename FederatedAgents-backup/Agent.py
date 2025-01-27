import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from collections import deque
import random
import os
from Dueling_Model import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#import torch
#import torch.nn as nn
#from itertools import count
import torch.nn.functional as F
#from tensorboardX import SummaryWriter



# --------------------------------------------------Dueling------------------------------------------------------------
class Dueling:
    def __init__(self, env, agent_id, config):
        self.env = env
        self.state_n = self.env.observation_space.shape[0]
        self.action_n = self.env.action_space.n
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.onlineQNetwork = QNetwork(self.state_n, self.action_n).to(self.device)
        self.targetQNetwork = QNetwork(self.state_n, self.action_n).to(self.device)
        self.targetQNetwork.load_state_dict(self.onlineQNetwork.state_dict())
        self.optimizer = torch.optim.Adam(self.onlineQNetwork.parameters(), lr=1e-4)
        self.gamma = float(config.get('DRL', 'gamma'))
        self.EXPLORE = 20000
        #self.INITIAL_EPSILON = 0.1
        self.INITIAL_EPSILON = 0.1
        self.FINAL_EPSILON = 0.00001
        self.REPLAY_MEMORY = 50000
        self.BATCH = 16
        self.learn_steps = 0
        self.UPDATE_STEPS = 4
        self.memory_replay = Memory(self.REPLAY_MEMORY)
        self.epsilon = self.INITIAL_EPSILON
        self.learn_steps = 0
        #self.writer = SummaryWriter('logs/ddqn')
        self.epsilon = int(config.get('DRL', 'epsilon'))
        self.epsilon_decay = float(config.get('DRL', 'epsilon_decay'))
        self.epsilon_min = float(config.get('DRL', 'epsilon_min'))
        

    def epsgreedyaction(self, state):
        #p = random.random()
        #if p < self.epsilon:
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.env.action_space.n)
        else:
            tensor_state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            #print("tensor_state-----------------------",tensor_state)
            action = self.onlineQNetwork.select_action(tensor_state)
            print("####################################I am in exploration-----------------------")
        return (action)
        
    def memoryrep(self,state, next_state, action, reward, done):
        self.memory_replay.add((state, next_state, action, reward, done))
        if self.memory_replay.size() > 128:
          '''
          if begin_learn is False:
              print('learn begin!')
              begin_learn = True
          '''
          self.learn_steps += 1
          if self.learn_steps % self.UPDATE_STEPS == 0:
              self.targetQNetwork.load_state_dict(self.onlineQNetwork.state_dict())
          batch = self.memory_replay.sample(self.BATCH, False)
          batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)
        
          batch_state = torch.FloatTensor(batch_state).to(self.device)
          batch_next_state = torch.FloatTensor(batch_next_state).to(self.device)
          batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(self.device)
          batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(self.device)
          batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(self.device)
        
          with torch.no_grad():
              onlineQ_next = self.onlineQNetwork(batch_next_state)
              targetQ_next = self.targetQNetwork(batch_next_state)
              online_max_action = torch.argmax(onlineQ_next, dim=1, keepdim=True)
              y = batch_reward + (1 - batch_done) * self.gamma * targetQ_next.gather(1, online_max_action.long())
        
          loss = F.mse_loss(self.onlineQNetwork(batch_state).gather(1, batch_action.long()), y)
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
          #writer.add_scalar('loss', loss.item(), global_step=learn_steps)
          
          '''
          if self.epsilon > self.FINAL_EPSILON:
              self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE
          '''
          
          
          if self.epsilon > self.epsilon_min:
              self.epsilon *= self.epsilon_decay  



# --------------------------------------------------DQN------------------------------------------------------------
class DQN:
    def __init__(self, env, agent_id, config):
        self.env = env
        self.code_model = config.get('DRL', 'code_model')
        self.agent_id = agent_id
        self.gamma = float(config.get('DRL', 'gamma'))
        self.epsilon = int(config.get('DRL', 'epsilon'))
        self.epsilon_decay = float(config.get('DRL', 'epsilon_decay'))
        self.epsilon_min = float(config.get('DRL', 'epsilon_min'))
        self.learning_rate = float(config.get('DRL', 'learning_rate'))
        self.buffer_size = int(config.get('DRL', 'buffer_size'))
        self.best_score = -np.Inf
        self.best_score_FDRL = -np.Inf
        self.best_d_traffic = -np.Inf
        self.best_b_latency = -np.Inf
        self.best_o_channel = -np.Inf
        self.best_average_PRBs = -np.Inf
        self.PRB_buffer = []
        self.replay_memory = deque(maxlen=(int(config.get('DRL', 'replay_memory_size'))))
        self.train_network = self.Policy_Network()

        # Detect GPU for accelerating the computation
        gpus = tf.config.experimental.list_physical_devices('GPU')

        if len(gpus) != 0:
            self.device = '/gpu:0'
        else:
            self.device = '/cpu:0'

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    ##############################################################################
    def Policy_Network(self):
        """ Creates a dense network to estimate a policy """

        model = models.Sequential()
        model.add(Dense(24, input_dim=self.env.observation_space.shape[0], activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss=self._huber_loss, optimizer=Adam(learning_rate=self.learning_rate))
        return model

    ##############################################################################
    def epsgreedyaction(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.env.action_space.n)
        act_values = self.train_network.predict(state)
        return np.argmax(act_values[0])  # returns action

    ##############################################################################
    def train_buffer(self):
        if len(self.replay_memory) < self.buffer_size:
            return
        """ Dataset based optimization using Tensorflow """
        minibatch = random.sample(self.replay_memory, self.buffer_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.train_network.predict(next_state)[0]))
            target_f = self.train_network.predict(state)
            target_f[0][action] = target
            self.train_network.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    ##############################################################################
    def update_agent_network(self, weights):
        self.train_network.set_weights(weights)


# --------------------------------------------------DDQN------------------------------------------------------------
class DDQN:
    def __init__(self, env, agent_id, config):

        self.env = env
        self.code_model = config.get('DRL', 'code_model')
        self.agent_id = agent_id
        self.gamma = float(config.get('DRL', 'gamma'))
        self.epsilon = int(config.get('DRL', 'epsilon'))
        self.epsilon_decay = float(config.get('DRL', 'epsilon_decay'))
        self.epsilon_min = float(config.get('DRL', 'epsilon_min'))
        self.learning_rate = float(config.get('DRL', 'learning_rate'))
        self.buffer_size = int(config.get('DRL', 'buffer_size'))
        self.best_score = -np.Inf
        # self.best_score_per_step = -np.Inf
        self.best_score_FDRL = -np.Inf
        self.best_d_traffic = -np.Inf
        self.best_b_latency = -np.Inf
        self.best_o_channel = -np.Inf
        self.best_average_PRBs = -np.Inf
        self.best_traffic_tracing = -np.Inf
        self.PRB_buffer = []
        self.replay_memory = deque(maxlen=(int(config.get('DRL', 'replay_memory_size'))))
        self.train_network = self.Policy_Network()
        self.target_network = self.Policy_Network()
        self.target_network.set_weights(self.train_network.get_weights())

        # Detect GPU for accelerating the computation
        gpus = tf.config.experimental.list_physical_devices('GPU')

        if len(gpus) != 0:
            self.device = '/gpu:0'
        else:
            self.device = '/cpu:0'

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    ##############################################################################
    def Policy_Network(self):
        model = models.Sequential() #linear stack of layers https://keras.io/models/sequential/
        model.add(Dense(24, input_dim=self.env.observation_space.shape[0], activation='relu')) #[Input] -> Layer 1
        #   Dense: Densely connected layer https://keras.io/layers/core/
        #   24: Number of neurons
        #   input_dim: Number of input variables
        #   activation: Rectified Linear Unit (relu) ranges >= 0
        model.add(Dense(24, activation='relu')) #Layer 2 -> 3
        model.add(Dense(self.env.action_space.n, activation='linear')) #Layer 3 -> [output]
        #   Size has to match the output (different actions)
        #   Linear activation on the last layer
        model.compile(loss='mean_squared_error', #Loss function: Mean Squared Error
                      optimizer=Adam(learning_rate=self.learning_rate)) #Optimaizer: Adam (Feel free to check other options)
        return model
    
    
    
    '''
    def Policy_Network(self):
        """ Creates a dense network to estimate a policy """

        model = models.Sequential()
        #model.add(Dense(24, input_dim=self.env.observation_space.shape[0], activation='relu'))
        #model.add(Dense(24, activation='relu'))
        model.add(Dense(2, input_dim=self.env.observation_space.shape[0], activation='relu'))
        model.add(Dense(2, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss=self._huber_loss, optimizer=Adam(learning_rate=self.learning_rate))
        return model
    '''
    ##############################################################################
    def epsgreedyaction(self, state):
        """ Agent picks an action based on Epsilon-Greedy Policy """

        if self.code_model == 'Train':

            #self.epsilon = max(self.epsilon_min, self.epsilon)
            if np.random.rand() <= self.epsilon:
            #if np.random.rand(1) < self.epsilon:
                action = random.randrange(self.env.action_space.n) #self.env.action_space.sample()
            else:
                print("state in expoitation:----------------------------",state)
                action = np.argmax(self.train_network.predict(state)[0])

        elif self.code_model == 'Inference':
            action = np.argmax(self.train_network.predict(state)[0])

        return action

    ##############################################################################
    def train_buffer(self):
        """ Dataset based optimization using Tensorflow """
        if len(self.replay_memory) < self.buffer_size:
            return

        samples = random.sample(self.replay_memory, self.buffer_size)

        states = []
        new_states = []
        for sample in samples:
            state, action, reward, new_state, done = sample
            states.append(state)
            new_states.append(new_state)

        states = np.array(states)
        new_states = np.array(new_states)

        states = states.reshape(self.buffer_size, self.env.observation_space.shape[0])
        new_states = new_states.reshape(self.buffer_size, self.env.observation_space.shape[0])

        targets = self.train_network.predict(states)
        new_state_targets = self.target_network.predict(new_states)

        for i, sample in enumerate(samples):
            state, action, reward, new_state, done = sample
            target = targets[i]
            if done:
                target[action] = reward
            else:
                Q_future = max(new_state_targets[i])
                target[action] = reward + Q_future * self.gamma

        tf.debugging.set_log_device_placement(True)
        with tf.device(self.device):
            self.train_network.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return

##############################################################################
    def update_agent_network(self, weights):
        self.train_network.set_weights(weights)
        self.target_network.set_weights(weights)

