from __future__ import division
# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import random
import game
import util
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#implementation of framework of target net and evaluation net
class Net(nn.Module):
    def __init__(self, action_num, feature_num):
        super(Net, self).__init__()
        #the first fully connect layer
        self.fc1 = nn.Linear(feature_num, 50)
        #randomly initialize the fc layer
        self.fc1.weight.data.normal_(0,0.1)
        # the second fully connect layer
        self.fc2 = nn.Linear(50, 25)
        # randomly initialize the fc layer
        self.fc2.weight.data.normal_(0, 0.1)
        #third layer
        self.fc3 = nn.Linear(25,10)
        self.fc3.weight.data.normal_(0,0.1)
        #out_layer
        self.out = nn.Linear(10, action_num)
        self.out.weight.data.normal_(0, 0.1)
    #forward function in pytorch module
    def forward(self,x):
        #first layer
        x = self.fc1(x)
        x = F.relu(x)

        #second layer
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        #x = self.drop1(x)
        x = F.relu(x)
        #output layer
        actions_value = self.out(x)
        # actions_value = F.softmax(actions_value)
        return actions_value
# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.01, epsilon=0.2, gamma=0.9, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        #additional hyperparameters needed for DQN
        self.batch_size = 512
        #the update rate of the target network
        self.target_replace_iter = 300
        #capacity of the replay memory
        self.memory_capacity = 3000
        #number of actions that the pacman could take
        self.action_num = 4
        #number of the features that the enriorment could provide
        self.feature_num = None
        #state cache
        self.state_cache = None
        #score cache
        self.score_cache = 0
        #action cache
        self.action_cache = None

        #For the DQN:
        #define the evaluation and target network
        self.eval_net = None
        self.target_net = None
        self.learn_step_counter = None  # for the timing of target net update
        self.memory_counter = 0  # counter of memory
        # initialize the memory
        self.memory = None
        # optimizer of the DQN
        self.optimizer = None
        # loss function
        self.loss_func = nn.MSELoss(reduction='mean')
        #a gate to start the initilization of DQN
        self.game_start = 0
        #state that the game is on:1 end:0
        self.game_state = 1
        self.loss = 0
    #function that initialize the parameters for the DQN
    def DQN_INIT(self,action_num, feature_num,memory_capacity, LR, epsilon, batch_size,target_replace_iter,gamma):
        self.eval_net, self.target_net = Net(action_num, feature_num), Net(action_num, feature_num)

        self.learn_step_counter = 0 #for the timing of target net update
        self.memory_counter = 0 #counter of memory
        #initialize the memory
        self.memory = np.zeros((memory_capacity,feature_num*2+2))
        #optimizer of the DQN
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),lr = LR)
        #loss function
        self.loss_func = nn.MSELoss()
        #epsilon
        self.epsilon = epsilon
        #action_num
        self.action_num = action_num
        #feature_num
        self.feature_num = feature_num
        #memory capacity
        self.memory_capacity = memory_capacity
        #batch size
        self.batch_size = batch_size
        #target_replace_iter
        self.target_replace_iter = target_replace_iter
        #gamma
        self.gamma = 0.9

    # CHOOSE the action according to the enviroment features and legal moves
    def choose_action(self, x, legal_move, env_list):
        # expand dimension
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # only input one sample here
        # greedy agent
        action_sort = []
        if np.random.uniform() > self.epsilon:
            actions_value = self.eval_net.forward(x)

            # return  the sort index of the value
            sort_idx = torch.sort(actions_value, 1, descending=True)[1].data.numpy()[0]
            #only leave the legal move
            for i in range(4):
                if (sort_idx[i] in legal_move):
                    action_sort.append(sort_idx[i])
            action = action_sort[0]
        # random agent
        else:
            action = np.random.choice(legal_move)
        if (env_list[0:2]==[5,3]) and (env_list[12]==-1) and(env_list[28]==1):
            return 3
        return action

    # store the memory of the state, reward, action and new_state
    def memory_store(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # if the memory is full, over write the old values
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    # update the target_net and the learn from the memory
    def learn(self):
        # update the parameters in the target net
        if self.learn_step_counter % self.target_replace_iter == 0:
            #print('updating target network')
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # extract batch size examples from the memory
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_s = torch.FloatTensor(batch_memory[:, :self.feature_num])
        batch_a = torch.LongTensor(batch_memory[:, self.feature_num:self.feature_num + 1].astype(int))
        batch_r = torch.FloatTensor(batch_memory[:, self.feature_num + 1:self.feature_num + 2])
        batch_s_ = torch.FloatTensor(batch_memory[:, -self.feature_num:])

        # for the actions batch_a, pick the q_evalation value
        q_eval = self.eval_net(batch_s).gather(1, batch_a)  # shape (batch, 1)
        # we don't want to learn the the target net during the training, it only copied from the evaluation_net
        q_next = self.target_net(batch_s_).detach()
        q_target = batch_r + self.gamma * (q_next.max(1)[0].view(self.batch_size,1)) * self.game_state # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        self.loss = loss
        # now compute and update the evaluation net, following the rules of torch
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
            return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value
        
    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    #function to reset the caches after the end of one game
    def reset_caches(self):
        self.action_cache = None
        self.state_cache = None
        self.score_cache = 0
        self.game_state = 1

    #turn the information of the encironment into list of features(input of network)
    def get_state(self,pac_pos, ghost_pos, food_loc):

        env_list = list(pac_pos)
        env_list.extend(list(ghost_pos[0]))

        #set the F to be -1 and T to be 1 in posotion feature
        for i in list(food_loc):
            for j in i:
                if j == False:
                    env_list.append(-1)
                else:
                    env_list.append(1)

        return env_list

    #convert the move list to a number list
    def turn_move_to_list(self,legal_move):
        legal_move_list = list(legal_move)
        num_list = []
        #north:0 south: 1 east:2 west:3
        for direction in legal_move_list:
            if direction == 'North':
                num_list.append(0)
            if direction == 'South':
                num_list.append(1)
            if direction == 'East':
                num_list.append(2)
            if direction == 'West':
                num_list.append(3)
        return num_list

    #convert number to move
    def turn_number_to_move(self,action):
        if action == 0:
            return 'North'
        if action == 1:
            return 'South'
        if action == 2:
            return 'East'
        if action == 3:
            return 'West'
        if action == 4:
            return 'Stop'

    # convert move to number
    def turn_move_to_num(self, action):
        if action == 'North':
            return 0
        if action == 'South':
            return 1
        if action == 'East':
            return 2
        if action == 'West':
            return 3

    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):

        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # GET the current state
        env_list = self.get_state(state.getPacmanPosition(), state.getGhostPositions(), state.getFood())
        # get the feature number
        if self.feature_num == None:
            self.feature_num = len(env_list)

        #initilize the DQN network when we start to play
        if self.game_start is 0:
            self.DQN_INIT(self.action_num, self.feature_num,self.memory_capacity,self.alpha,self.epsilon,
                          self.batch_size,self.target_replace_iter,self.gamma)
        self.game_start = 1

        #get the list of legal move
        legal_move_list = self.turn_move_to_list(legal)

        #choose the next action, if we just start the game, pick the first move randomly
        if self.action_cache == None:
            action = random.choice(legal_move_list)
        else:
            #action = random.choice(self.turn_move_to_number(legal))
            action = self.choose_action(env_list,legal_move_list,env_list)


        #store the s,a,r,s_ into memory
        reward = (state.getScore()-self.score_cache)

        if self.state_cache is not None:
            #eat the pac in the middle first would not give a positive reward
            if (reward==9) and (env_list[12]==1) and(env_list[28]==-1):
                for i in range(40):
                    self.memory_store(self.state_cache, self.action_cache, (-0.05*reward),env_list)
            elif (env_list[12]==-1) and(env_list[28]==1) and(reward==9):
                for i in range(20):
                    self.memory_store(self.state_cache, self.action_cache, (0.1*reward),env_list)
            else:
                self.memory_store(self.state_cache, self.action_cache, (0.1* reward), env_list)

        #when the memory is full, let the DQN network learn
        if self.memory_counter > self.memory_capacity:
            self.learn()
            
        #you can print the loss if you want to see it
        if self.memory_counter % 500 == 0:
            #print(env_list)
            print(self.memory_counter)
            print('loss: %s' % self.loss)

        #store the new state and reward and action into the cache
        self.state_cache = env_list
        self.score_cache = state.getScore()
        self.action_cache = action

        # Now pick what action to take. For now a random choice among
        # the legal moves
        pick = self.turn_number_to_move(action)
        # We have to return an action
        return pick
            

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):

        self.game_state = 0
        #print "A game just ended!"
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        # GET the current state(win or loss)
        env_list = self.get_state(state.getPacmanPosition(), state.getGhostPositions(), state.getFood())

        # store the s,a,r,s_ into memory
        reward = (state.getScore() - self.score_cache)

        if self.state_cache is not None:
            if reward < -10:
                for i in range(30):
                    self.memory_store(self.state_cache, self.action_cache, (0.1 * reward), np.zeros_like(self.state_cache))
            else:
                for i in range(25):
                    self.memory_store(self.state_cache, self.action_cache, 0.1 * reward, np.zeros_like(self.state_cache))

        # when the memory is full, let the DQN network learn
        if self.memory_counter > self.memory_capacity:
            self.learn()

        #reset the caches after the end of a game
        self.reset_caches()

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)



