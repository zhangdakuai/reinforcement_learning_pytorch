#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
#######################################################################
# Copyright (C)                                                       #
# 2016 - 2019 Pinard Liu(liujianping-ok@163.com)                      #
# https://www.cnblogs.com/pinard                                      #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
##https://www.cnblogs.com/pinard/p/9714655.html ##
## 强化学习（八）价值函数的近似表示与Deep Q-Learning ##

#######################################################################
# Copyright (C) 2022 #
# @Time    : 2022/7/9 9:23 下午
# @Author  : 张大快(zhangdakuai)
# @Email   : dakuaizhang@gmail.com
# @File    : deep_q_learning.py
#######################################################################




import gym
import torch
import numpy as np
import random
from collections import deque
import torch.nn.functional as F
from torchviz import make_dot

# Hyper Parameters for DQN
GAMMA = 0.95 # discount factor for target Q
INITIAL_EPSILON = 0.2 # starting value of epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch


class QNetwork(torch.nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        # init some parameters
        self.time_step = 0
        self.state_dim, self.action_dim = env.observation_space.shape[0], env.action_space.n
        self.H = 20
        self.linear1 = None
        self.linear2 = None
        self.q_value = None
        self.optimizer = None

        self.loss = None
        self.init_q_network()
        self.set_optimizer()

    def init_q_network(self):
        """
        create softmax network
        """
        self.linear1 = torch.nn.Linear(self.state_dim, self.H)
        self.linear2 = torch.nn.Linear(self.H, self.action_dim)

    def forward(self, x):
        """
        forward func , 定义前向传播
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        self.q_value = y_pred
        return self.q_value

    def set_optimizer(self):
        """
        set optimizer
        @return:
        """
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.85)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.optimizer.zero_grad()

    def my_loss(self, state_batch, action_input, y_batch):
        """
        self define loss function
        @param state_batch:
        @param action_input:
        @param y_batch:
        @return:
        """
        q_value = self.forward(state_batch)
        q_action = torch.sum(q_value * action_input, dim=1)
        mseloss = torch.nn.MSELoss(reduction='mean')
        self.loss = mseloss(q_action, y_batch)

class DeepQLearning():
    def __init__(self, env):
        self.q_net = QNetwork(env)
        self.GAMMA = 0.95
        self.INITIAL_EPSILON = 0.5
        self.FINAL_EPSILON = 0.01
        self.epsilon = self.INITIAL_EPSILON
        self.REPLAY_SIZE = 1000
        self.BATCH_SIZE = 25
        self.replay_memory = deque()
        self.time_step = 0
        # init q_network parameters
        torch.nn.init.normal_(self.q_net.linear1.weight)
        torch.nn.init.constant_(self.q_net.linear1.bias, 0.01)
        torch.nn.init.normal_(self.q_net.linear2.weight)
        torch.nn.init.constant_(self.q_net.linear2.bias, 0.01)
        self.sample_cnt = 0

    def store_and_train(self, state, action, reward, next_state, done):
        """
        push sample in queue & call train Q_network
        @param state:
        @param action:
        @param reward:
        @param next_state:
        @param done:
        @return:
        """
        self.sample_cnt += 1
        one_hot_action = np.zeros(self.q_net.action_dim)
        one_hot_action[action] = 1
        self.replay_memory.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_memory) > REPLAY_SIZE:
            self.replay_memory.popleft()

        if len(self.replay_memory) > BATCH_SIZE :# and self.sample_cnt % 5 == 0 :
            self.train_q_network()

    def train_q_network(self):
        """
        train q network
        @return:
        """
        self.time_step += 1

        data_batch = random.sample(self.replay_memory, self.BATCH_SIZE)
        state_batch = [data[0] for data in data_batch]
        action_batch = [data[1] for data in data_batch]
        reward_batch = [data[2] for data in data_batch]
        next_state_batch = [data[3] for data in data_batch]

        #
        y_batch = []
        q_value_batch = self.q_net(torch.tensor(np.array(next_state_batch)))
        q_value_batch = q_value_batch.detach().numpy()

        for i in range(0, self.BATCH_SIZE):
            done = data_batch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.GAMMA * np.max(q_value_batch[i]))

        # 计算损失
        self.q_net.my_loss(torch.tensor(np.array(state_batch))
                                        , torch.tensor(np.array(action_batch))
                                        , torch.tensor(np.array(y_batch)))
        # 反向传播, 更新参数
        self.q_net.optimizer.zero_grad()
        self.q_net.loss.backward()
        self.q_net.optimizer.step()

    def egreedy_action(self, state):
        q_value = self.q_net(torch.tensor(np.array(state)))
        if random.random() <= self.epsilon:
            self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / 10000
            return random.randint(0, self.q_net.action_dim - 1)
        else:
            self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / 10000
            return np.argmax(q_value.detach().numpy())

    def action(self, state):
        """
        given a state ,return the aciton
        @param state:
        @return:
        """
        q_value = self.q_net(torch.tensor(np.array(state)))
        return np.argmax(q_value.detach().numpy())

    def choose_action(self, observation):
        """
        预估当前状态observation下的动作概率分布
        @param observation:
        @return:
        """
        # 预估当前状态下的动作概率分布
        prob_weights = self.q_net(torch.tensor(np.array([observation]) ))
        prob_weights = prob_weights.detach().numpy()
        # 选择一个动作
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

class MyAgent():
    def __init__(self, env_name, episode=1000, step=100,test_size=20):
        self.env = gym.make(env_name)
        self.EPISODE = episode
        self.STEP = step
        self.TEST_SIZE = test_size
        self.agent = DeepQLearning(self.env)

    def run(self):

        for episode in range(self.EPISODE):
            # initialize task
            state = self.env.reset()
            # Train
            for step in range(self.STEP):
                action = self.agent.egreedy_action(state)  # e-greedy action for train
                next_state, reward, done, _ = self.env.step(action)
                # Define reward for agent
                reward = -1.0 if done else 0.1
                self.agent.store_and_train(state, action, reward, next_state, done)
                state = next_state
                if done and episode % 100 == 0:
                    loss = 0
                    if self.agent.q_net.loss is not None:
                        loss = self.agent.q_net.loss.item()
                    print(f"[main]iter for {episode}, {step},len={len(self.agent.replay_memory)} state={state}" +
                          f" , action = {action} ,loss ={loss}")
                    # agent.learn()
                    # state = env.reset()
                    break

            # Test every 100 episodes
            if episode % 100 == 0:
                total_reward = 0
                for i in range(self.TEST_SIZE):
                    state = self.env.reset()
                    for j in range(self.TEST_SIZE):
                        self.env.render()
                        action = self.agent.action(state)  # direct action for test
                        state, reward, done, _ = self.env.step(action)
                        total_reward += reward
                        if done:
                            break
                ave_reward = total_reward / self.TEST_SIZE
                print(f"episode: {episode} , Evaluation Average Reward: {ave_reward},reward = {reward}")
                # break
        state_t = torch.tensor(np.array(state))
        p = self.agent.q_net(state_t)
        # make_dot(p, params=dict(list(agent.q_net.parameters()))).render("dpn_torch",format="png")
        input_names = ['state']
        out_put_name = ['q_value']
        export_file = './pic/dqn_torch.onnx'
        torch.onnx.export(self.agent.q_net, state_t, export_file, input_names=input_names, output_names=out_put_name)

if __name__ == '__main__':
    ENV_NAME = 'CartPole-v1'
    EPISODE = 1000  # Episode limitation 100个回合
    STEP = 256  # Step limitation in an episode ，每个回合，执行100次策略
    TEST = 20  # The number of experiment test every 100 episode

    agent = MyAgent(env_name=ENV_NAME, episode=EPISODE, step=STEP, test_size=TEST)
    agent.run()
