# coding:utf-8
import os
import sys
import argparse
from turtle import done
import numpy as np
import gym
import highway_env
# from random import sample
# from CarlaLCEnv import CarlaEnv, PlayGame

import matplotlib.pyplot as plt
import copy
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from tensorboardX import SummaryWriter
import pickle
from models import *
import random
from network import Dueling_DQN, Dueling_DQN_vector, attention_Dueling_DQN, EgoAttentionNetwork, MultiAttentionNetwork
from torch.autograd import Variable
import pdb
from numpy import *
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["SDL_VIDEODRIVER"] ="dummy"
# hyper-parameters
render_op = False


class ReplayBuffer(object):
    def __init__(self, memory_size=5000):
        self.buffer = []
        self.memory_size = memory_size
        self.next_idx = 0

    def push(self, state, action, reward, next_state, done):
        if isinstance(action, np.ndarray):
            action = action[0]

        data = (state, action, reward, next_state, done)
        if len(self.buffer) <= self.memory_size:
            self.buffer.append(data)
        else:
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, self.size() - 1)
            data = self.buffer[idx]
            state, action, reward, next_state, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        # pdb.set_trace()
        # pdb.set_trace()
        if isinstance(states[0], tuple):
            return states, np.array(actions), np.array(rewards), next_states, np.array(dones)
        else:
            return torch.cat(states), np.array(actions), np.array(rewards), torch.cat(next_states), np.array(dones)

    def size(self):
        return len(self.buffer)


class DDQN(object):
    def __init__(self, num_states, num_actions, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_actions = num_actions
        self.args = args
        self._init_hyperparameters()
        if args.input == 'image':
            self.eval_net, self.target_net = Dueling_DQN(num_states, num_actions, args.add_dueling).to(
                self.device), Dueling_DQN(num_states, num_actions, args.add_dueling).to(self.device)
        if args.input == 'image_attention':
            self.eval_net, self.target_net = attention_Dueling_DQN(num_states, num_actions, args.add_dueling).to(
                self.device), attention_Dueling_DQN(num_states, num_actions, args.add_dueling).to(self.device)
        if args.input == 'vector':
            self.eval_net, self.target_net = Dueling_DQN_vector(num_states, num_actions, args.add_dueling).to(
                self.device), Dueling_DQN_vector(num_states, num_actions, args.add_dueling).to(self.device)
        if args.input == 'vector_attention':
            self.eval_net, self.target_net = EgoAttentionNetwork(num_states, num_actions, args.add_dueling).to(
                self.device), EgoAttentionNetwork(num_states, num_actions, args.add_dueling).to(self.device)
        if args.input == 'multi_attention':
            self.eval_net, self.target_net = MultiAttentionNetwork(num_states[0], num_states[1], num_actions,
                                                                   args.add_dueling).to(
                self.device), MultiAttentionNetwork(num_states[0], num_states[1], num_actions, args.add_dueling).to(
                self.device)
        print(self.eval_net)
        self.memory = ReplayBuffer()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.mean_reward = []

    def choose_action(self, state):

        if np.random.uniform() < self.epsilon:
            if isinstance(state, tuple):
                action_value = self.eval_net.forward(state[0].to(self.device), state[1].to(self.device))
            else:
                action_value = self.eval_net.forward(state.to(self.device))
            action = torch.max(action_value, 1)[1].cpu().data.numpy()
        else:
            action = np.array([np.random.randint(0, self.num_actions)])
        action = action[0]

        return action

    def learn(self):
        b_s, b_a, b_r, b_s_, b_done = self.memory.sample(self.batchsize)
        if isinstance(b_s[0], tuple):
            vec_s = torch.cat([b_s[i][0] for i in range(len(b_s))])
            vec_s = Variable(vec_s).to(self.device)
            img_s = torch.cat([b_s[i][1] for i in range(len(b_s))])
            img_s = Variable(img_s).to(self.device)
            vec_s_ = torch.cat([b_s_[i][0] for i in range(len(b_s_))])
            vec_s_ = Variable(vec_s_).to(self.device)
            img_s_ = torch.cat([b_s_[i][1] for i in range(len(b_s_))])
            img_s_ = Variable(img_s_).to(self.device)
        else:
            b_s = Variable(b_s).to(self.device)
            b_s_ = Variable(b_s_).to(self.device)
        b_a = Variable(torch.LongTensor(b_a)).reshape(self.batchsize, -1).to(self.device)
        b_r = Variable(torch.FloatTensor(b_r)).reshape(self.batchsize, -1).to(self.device)
        b_done = Variable(torch.BoolTensor(b_done)).reshape(self.batchsize, -1).to(self.device)
        if self.args.input == 'multi_attention':
            q_eval = self.eval_net(vec_s, img_s).gather(1, b_a)
            if self.args.double:
                q_next_action = self.eval_net(vec_s_, img_s_).max(1)[1].reshape(self.batchsize,                                                                              -1).detach()  # 如果这里是target就是传统的dqn  现在为double dqn
            else:
                q_next_action = self.target_net(vec_s_, img_s_).max(1)[1].reshape(self.batchsize, -1).detach()
            q_next = self.target_net(vec_s_, img_s_).gather(1, q_next_action)
        else:
            q_eval = self.eval_net(b_s).gather(1, b_a)
            if self.args.double:
                q_next_action = self.eval_net(b_s_).max(1)[1].reshape(self.batchsize,
                                                                      -1).detach()  # 如果这里是target就是传统的dqn  现在为double dqn
            else:
                q_next_action = self.target_net(b_s_).max(1)[1].reshape(self.batchsize,
                                                                        -1).detach()
            q_next = self.target_net(b_s_).gather(1, q_next_action)
        q_target = b_r + self.gamma * q_next
        # pdb.set_trace()

        q_target = torch.where(
            b_done.reshape(self.batchsize, -1), b_r, q_target
        )

        loss = self.loss_func(q_eval, q_target.detach())

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def save(self, directory, i):
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(self.eval_net.state_dict(), directory + 'dqn{}.pth'.format(i))

    def load(self, directory, i):
        self.eval_net.load_state_dict(torch.load(directory + 'dqn{}.pth'.format(i)))
        print("====================================")
        print("Model has been loaded...")
        print("====================================")

    def _init_hyperparameters(self):
        self.lr = 0.001
        self.batchsize = 64
        # self.replace_type = 'hard'
        self.tau = 0.005
        # self.var = 3
        self.epsilon_max = 0.999
        self.epsilon_min = 0.1
        self.eps_decay = self.args.epsilon_decay

        self.epsilon = self.epsilon_min

        self.gamma = 0.99
        # self.target_network_replace_freq = 50
        self.save_freq = 200
        self.seed = self.args.seed

        if self.seed != None:
            # Check if our seed is valid first
            assert (type(self.seed) == int)

            # Set the seed
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def observe(self, state):
        state_image = state['image']
        state_ki = state['kinematics']
        state_image = torch.from_numpy(state_image.transpose(2, 0, 1)[None] / 255).float()
        state_ki = torch.from_numpy(state_ki.reshape(1, -1)).float()
        if self.args.input == 'image':
            return state_image
        if self.args.input == 'image_attention':
            return state_image
        if self.args.input == 'vector':
            return state_ki
        if self.args.input == 'vector_attention':
            return torch.tensor([state['kinematics']], dtype=torch.float)
        if self.args.input == 'multi_attention':
            return (torch.tensor([state['kinematics']], dtype=torch.float), state_image)


def parse_args(args):
    # """ Parse arguments from command line input
    # """
    parser = argparse.ArgumentParser(description='Training parameters')
    # #
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'])  # mode = 'train' or 'test'
    parser.add_argument('--input', default='image', type=str,
                        choices=['image', 'vector', 'image_attention', 'vector_attention', 'multi_attention'])
    parser.add_argument('--file_name', default='./', type=str)
    parser.add_argument('--episode_num', default=1001, type=int)
    parser.add_argument('--epsilon_decay', default=6250, type=int)
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--gpu', type=str, default=0, help='GPU ID')
    parser.add_argument('--add_dueling', default=False, action='store_true')
    parser.add_argument('--double', default=False, action='store_true')
    parser.add_argument('--right_r', default=0.1, type=float)
    parser.add_argument('--highspeed_r', default=0.4, type=float)
    parser.add_argument('--normal', default=True, action='store_false')
    parser.add_argument('--absolute', default=False, action='store_true')
    parser.add_argument('--model_ep',default = 1000,type = int)

    return parser.parse_args(args)


def train(args=None):
    if not os.path.exists(args.file_name):
        os.mkdir(args.file_name)

    # Check if a GPU ID was set
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # world, client = PlayGame.setup_world(host='localhost', fixed_delta_seconds=0.05, reload=True)
    # # client.set_timeout(5.0)
    # if world is None:
    #     return
    # traffic_manager = client.get_trafficmanager(8000)
    env_config = {
        "id": "highway-v0",
        "import_module": "highway_env",
        "lanes_count": 3,
        "vehicles_count": 50,
        "duration": 50,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        # "observation": {
        #     "type": "Kinematics",
        #     "vehicles_count": 15,
        #     "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        #     # "features_range": {
        #     #     "x": [-100, 100],
        #     #     "y": [-100, 100],
        #     #     "vx": [-20, 20],
        #     #     "vy": [-20, 20]
        #     # },
        #     # "absolute": True,
        #     "order": "shuffled"
        # },
        "observation": {
            "type": "GrayscaleAndKinematics",
            "observation_shape": (150, 600),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.,
            # "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "order": "shuffled",
            "normalize" : args.normal,
            "absolute": args.absolute
        },

        "offscreen_rendering": True,
        "screen_width": 600,  # [px]
        "screen_height": 150,  # [px]
        "RIGHT_LANE_REWARD": args.right_r,
        "HIGH_VELOCITY_REWARD": args.highspeed_r,

        # "destination": "o1"
    }
    env = gym.make("highway-v0")
    env.unwrapped.configure(env_config)
    
    #pdb.set_trace()

    # directory = './weights_with_ego_attention/'
    # dqn.writer = SummaryWriter(directory)
    episodes = args.episode_num
    # print("Collecting Experience....")
    mean_reward_list = []
    reward_list = []
    if args.input == 'image':
        agent = DDQN(np.array([4, 150, 600]), 5, args)
    if args.input == 'image_attention':
        agent = DDQN(np.array([4, 150, 600]), 5, args)
    if args.input == 'vector':
        agent = DDQN(15 * 7, 5, args)
    if args.input == 'vector_attention':
        agent = DDQN(7, 5, args)
    if args.input == 'multi_attention':
        agent = DDQN([7, [4, 150, 600]], 5, args)

    t_so_far = 0
    ep_r_be = 0

    for i in range(episodes):
        ep_r = 0
        ep_t = 0
        s = env.reset()
        s = agent.observe(s)
        if render_op:
            env.render()
        # pdb.set_trace()
        while True:
            if t_so_far % 200 == 0:
                agent.target_net.load_state_dict(agent.eval_net.state_dict())
            agent.epsilon = agent.epsilon_max - (agent.epsilon_max - agent.epsilon_min) * np.exp(
                -1.0 * t_so_far / agent.eps_decay)
            t_so_far += 1
            a = agent.choose_action(s)
            s_, r, done, info = env.step(a)
            s_ = agent.observe(s_)

            # pdb.set_trace()

            if (info['crashed'] and done):
                agent.memory.push(s, a, r, s_, True)
            else:
                agent.memory.push(s, a, r, s_, False)
            # agent.memory.push(s,a,r,s_,done)
            ep_r += r

            if t_so_far > 50:
                loss = agent.learn()

            if done:
                reward_list.append(ep_r)
                if i == 0:
                    ep_r_be = ep_r
                ep_r = ep_r_be * 0.8 + ep_r * 0.2
                ep_r_be = ep_r
                mean_reward_list.append(ep_r)
                print('Ep:', i, '|', 'Ep_r:', round(ep_r, 2), '|', 't_so_far:', {t_so_far}, '|', 'epsilon:',
                      round(agent.epsilon, 3), 'max:', max(mean_reward_list))
                break
            s = s_

            if render_op:
                env.render()

        if i % 100 == 0:
            path_name = args.file_name + 'weights_with_{}'.format(args.input) + '/'
            agent.save(path_name, i)
            plt.plot(mean_reward_list)
            plt.savefig(path_name + 'mean_reward.png')
            plt.close()
            plt.plot(reward_list)
            plt.savefig(path_name + 'reward.png')
            plt.close()


def test(args=None):
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    env_config = {
        "id": "highway-v0",
        "import_module": "highway_env",
        "lanes_count": 3,
        "vehicles_count": 50,
        "duration": 50,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        # "observation": {
        #     "type": "Kinematics",
        #     "vehicles_count": 15,
        #     "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        #     # "features_range": {
        #     #     "x": [-100, 100],
        #     #     "y": [-100, 100],
        #     #     "vx": [-20, 20],
        #     #     "vy": [-20, 20]
        #     # },
        #     # "absolute": True,
        #     "order": "shuffled"
        # },
        "observation": {
            "type": "GrayscaleAndKinematics",
            "observation_shape": (150, 600),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.,
            # "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "order": "shuffled",
            "normalize" : args.normal,
            "absolute": args.absolute
        },

        "offscreen_rendering": True,
        "screen_width": 600,  # [px]
        "screen_height": 150,  # [px]
        "RIGHT_LANE_REWARD": 0.1,
        "HIGH_VELOCITY_REWARD": 0.4,
        # "destination": "o1"
    }
    env = gym.make("highway-v0")
    env.unwrapped.configure(env_config)
    # directory = './weights_with_ego_attention/'
    # dqn.epsilon = 0
    # dqn.load(directory, 1000)

    # 编写测试过程
    if args.input == 'image':
        agent = DDQN(np.array([4, 150, 600]), 5, args)
    if args.input == 'image_attention':
        agent = DDQN(np.array([4, 150, 600]), 5, args)
    if args.input == 'vector':
        agent = DDQN(15 * 7, 5, args)
    if args.input == 'vector_attention':
        agent = DDQN(7, 5, args)
    if args.input == 'multi_attention':
        agent = DDQN([7, [4, 150, 600]], 5, args)

    agent.epsilon = 1.0
    path_name = args.file_name + 'weights_with_{}'.format(args.input) + '/'
    agent.load(path_name, args.model_ep)
    # t_so_far = 0
    time_safe = 0
    ep_r_list = []
    lane_change_list = []
    step_list = []
    
    with open(args.file_name + 'result.txt', 'a') as result:
        result.write("load model episode: {}".format(args.model_ep))
        result.write('\n')

    for _ in range(10):
        state = env.reset()
        state = agent.observe(state)
        # state = state.flatten()
        # for _ in range(5):
        #     _, _, done = env.step(0)
        ep_reward = 0
        lane_change = 0
        for t in count():
            action = agent.choose_action(state)
            if action in [0, 2]:
                lane_change += 1
            next_state, reward, done, info = env.step(action)
            next_state = agent.observe(next_state)
            # next_state = next_state.flatten()
            ep_reward += reward
            if done:
                print("step: {}, ep_reward: {}".format(t, ep_reward))
                if info['crashed'] == False:
                    time_safe += 1
                ep_r_list.append(ep_reward)
                lane_change_list.append(lane_change)
                step_list.append(t + 1)
                with open(args.file_name + 'result.txt', 'a') as result:
                    result.write("step: {}, ep_reward: {}, lane change: {}".format(t, ep_reward, lane_change))
                    result.write('\n')
                break
            state = next_state
            # env.render()
    with open(args.file_name + 'result.txt', 'a') as result:
        result.write('average result:\n')
        result.write("step: {}, ep_reward: {}, lane change: {},safe:{}".format(mean(step_list), mean(ep_r_list),
                                                                               mean(lane_change_list), time_safe / 10))
        result.write('\n')


if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)
    if args.mode == 'train':
        train(args)
    else:
        test(args)
