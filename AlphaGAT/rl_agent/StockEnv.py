import gym
import matplotlib
import numpy as np
import pandas as pd
import torch
from gym.utils import seeding
import seaborn as sns
from scipy.special import softmax
from indicator.datasetstock import Dataset_Stock
from rl_agent.Metric import Metric

matplotlib.use("Agg")
import matplotlib.pyplot as plt

class  Stock_Env:
    def  __init__(self,
                  dataset: Dataset_Stock,
                  model,
                  stock_num,
                  state_dim,
                  G,
                  commission_ratio=0.0025,
                  allow_short=False,
                  initial=True,
                  mode="",
                  is_eval = False,
                  cwd = None
                  ):

        self.dataset = dataset
        self.model = model
        self.stock_num = stock_num
        self.state_dim = state_dim
        self.commission_ratio = commission_ratio
        self.allow_short = allow_short
        self.initial = initial
        self.mode =mode
        self.is_eval = is_eval
        self.cwd = cwd
        ####Number of stocks allowed for trading
        self.G =G
        self.day = 0
        self.max_days = len(self.dataset)
        # initialize reward
        self.reward = 0
        self.episode = 0
        self._seed()
        ###for th universal settting
        self.max_step = 123456789
        self.if_discrete = False
        self.print_verbosity = 1
        self._initiate_state()
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def reset(self):
        # initiate state
        self.day = 0
        self._initiate_state()
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.alpha_weight = []
        # self.rewards_memory.append(0.0)
        self.rewards_memory.append(1.0)
        self.actions_memory.append([1.0]+[0 for i in range(self.stock_num)])
        self.episode += 1
        ###((self.cache_day,self.stock_dim, self.cache_dim))
        return self.state

    def _initiate_state(self):
        row_data, reward_ratio = self.dataset.__getitem__(self.day)
        yt = reward_ratio.cpu().detach().numpy()/100.0 + 1
        yt = np.insert(yt, 0, 1.0)
        self.yt = yt
        last_yt = (row_data[-1,:,0] / row_data[-2, :, 0]).cpu().detach().numpy()
        last_yt = np.insert(last_yt, 0, 1.0)
        self.last_yt = last_yt

        #####
        self.last_weight = np.zeros((self.stock_num+1), dtype=float)
        self.last_weight[0] = 1.0
        #####generate alphas by model
        with torch.no_grad():
            self.model.eval()
            alphas = self.model(row_data.unsqueeze(0))
            mean = alphas.mean(dim=-1, keepdims=True)
            std = alphas.std(dim=-1, keepdims=True)
            normalized_alphas = (alphas - mean) / std
            normalized_alphas = normalized_alphas.detach().squeeze(0)
        np_state = alphas.detach().cpu().numpy()
        self.state = normalized_alphas
    def _update_state(self, portfilio_weight):
        row_data,reward_ratio  = self.dataset.__getitem__(self.day)
        self.last_yt = self.yt
        yt = reward_ratio.cpu().detach().numpy()/100.0 + 1
        yt = np.insert(yt, 0, 1.0)
        self.yt = yt
        self.last_weight =portfilio_weight
        #####alphas by model
        with torch.no_grad():
            self.model.eval()
            alphas = self.model(row_data.unsqueeze(0))
            mean = alphas.mean(dim=-1, keepdims=True)
            std = alphas.std(dim=-1, keepdims=True)
            normalized_alphas = (alphas - mean) / std
            normalized_alphas = normalized_alphas.detach().squeeze(0)
        self.state = normalized_alphas



    def generate_portfolio_weight(self, weight_alphas, stock_alphas):
        stock_scores = np.matmul(weight_alphas, stock_alphas).reshape((self.stock_num,))
        ####softmax
        sorted_indices = np.argsort(stock_scores)
        # Get the indices of the top five values
        k_largest_indices = sorted_indices[-self.G:]
        # Get the top five values
        top_five_values = stock_scores[k_largest_indices]
        # Apply softmax to the top five values
        softmax_values = softmax(top_five_values)
        portfilio_weight = np.zeros((self.stock_num + 1), dtype=float)
        i = 0
        for index in k_largest_indices:
            portfilio_weight[index+1] = softmax_values[i]
            i+=1;
        return portfilio_weight




    def step(self, weight_alphas):
        self.alpha_weight.append(weight_alphas)
        weight_alphas = weight_alphas[np.newaxis,:]
        ##weight_alphas[alphas, stocknum]
        stock_alphas = self.state.cpu().detach().numpy().reshape((-1 ,self.stock_num))
        portfilio_weight = self.generate_portfolio_weight(weight_alphas, stock_alphas)
        begin_weight = self.last_weight
        last_yt = self.last_yt
        yt = self.yt
        reward = np.dot(last_yt, begin_weight)
        future_omega = (last_yt * begin_weight) / reward
        pure_pc = 1 - np.sum(np.abs(begin_weight[1:] - future_omega[1:])) * self.commission_ratio
        next_day_reward = np.dot(yt, portfilio_weight)
        reward_final = next_day_reward * pure_pc
        self.reward = np.log(reward_final) *100
        self.rewards_memory.append(reward_final)
        self.actions_memory.append(portfilio_weight)
        self.day += 1
        self.terminal = self.day >= self.max_days
        if self.terminal:

            print(f"Episode: {self.episode}; Mode: {self.mode}")
            reward_array = np.array(self.rewards_memory)
            # result = np.prod(np.exp(reward_array))
            result = np.prod(reward_array)
            print(f"day: {self.day}, episode: {self.episode},  Mode: {self.mode}")
            print(f"total_reward: {result:0.2f}")
            print("===========================================")
            if self.is_eval:
                ####Calculate metrics.
                metric = Metric(reward_array)
                metric.calculate_print()
                #####  Create plots, including asset weight heatmaps and cumulative return charts.
                date = np.arange(len(self.actions_memory))
                day_return = np.cumprod(reward_array)
                actions_array = np.array(self.actions_memory)
                array_weight_alphas = np.array(self.alpha_weight)
                return_array = reward_array
                np.save(f"{self.cwd}/EnvFigs/returnArray_{self.episode}.npy", return_array)
                ####print action heat map
                action_data =pd.DataFrame(actions_array)
                plot = sns.heatmap(action_data)
                plt.savefig(f"{self.cwd}/EnvFigs/HEATMAP_{self.mode}_{self.episode}.jpg")
                plt.clf()
                plt.plot(date, day_return)
                plt.savefig(f"{self.cwd}/EnvFigs/RETURN_{self.mode}_{self.episode}.jpg")
                plt.close('all')

        else:
            self._update_state(portfilio_weight)
        return self.state, self.reward, self.terminal, {}
