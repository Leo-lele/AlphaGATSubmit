import os
import pickle

import gym
import torch
import numpy as np

class RLConfig:
    def __init__(self, agent_class=None, env_class=None, env_args=None):
        self.agent_class = agent_class  # agent = agent_class(...)
        self.if_off_policy = self.get_if_off_policy()  # whether off-policy or on-policy of DRL algorithm

        self.env_class = env_class  # env = env_class(**env_args)
        self.env_args = env_args  # env = env_class(**env_args)
        if env_args is None:  # dummy env_args  虚设的参数，防止编译错误产生的参数
            env_args = {'env_name': None, 'state_dim': None, 'action_dim': None, 'if_discrete': None}
        self.env_name = env_args['env_name']  # the name of environment. Be used to set 'cwd'.
        self.state_dim = env_args['state_dim']  # vector dimension (feature number) of state
        self.action_dim = env_args['action_dim']  # vector dimension (feature number) of action
        self.if_discrete = env_args['if_discrete']  # discrete or continuous action space
        self.num_envs = 1

        '''Arguments for reward shaping'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 1.0  # an approximate target reward usually be closed to 256

        '''Arguments for training'''
        self.net_dims = (64, 32)  # the middle layer dimension of MLP (MultiLayer Perceptron)
        self.learning_rate = 6e-5  # 2 ** -14 ~= 6e-5
        self.clip_grad_norm = 3.0  # 0.1 ~ 4.0, clip the gradient after normalization
        self.state_value_tau = 1e-3  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
        self.soft_update_tau = 5e-4  # 2 ** -8 ~= 5e-3. the tau of soft target update `net = (1-tau)*net + tau*net1`

        self.lambda_gae_adv = 0.95
        self.lambda_entropy = 0.01

        if self.if_off_policy:  # off-policy
            self.batch_size = int(64)  # num of transitions sampled from replay buffer.
            self.horizon_len = int(512)  # collect horizon_len step while exploring, then update network
            self.buffer_size = int(1e6)  # ReplayBuffer size. First in first out for off-policy.
            self.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
        else:  # on-policy
            self.batch_size = int(128)  # num of transitions sampled from replay buffer.
            self.horizon_len = int(2000)  # collect horizon_len step while exploring, then update network
            self.buffer_size = None  # ReplayBuffer size. Empty the ReplayBuffer for on-policy.
            self.repeat_times = 8.0  # repeatedly update network using ReplayBuffer to keep critic's loss small

        '''Arguments for device'''
        self.gpu_id = int(0)  # `int` means the ID of single GPU, -1 means CPU
        self.thread_num = int(8)  # cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`
        self.random_seed = int(0)  # initialize random seed in self.init_before_training()

        '''Arguments for evaluate'''
        self.cwd = None  # current working directory to save model. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = np.inf  # break training if 'total_step > break_step'
        self.break_score = np.inf  # break training if `cumulative_rewards > break_score`
        self.if_keep_save = True  # keeping save the checkpoint. False means save until stop training.
        self.if_over_write = True  # overwrite the best policy network. `self.cwd/actor.pth`
        self.if_save_buffer = False  # if save the replay buffer for continuous training after stop training

        self.save_gap = int(8)  # save actor f"{cwd}/actor_*.pth" for learning curve.
        self.eval_times = 1  # number of times that get the average episodic cumulative return
        self.eval_per_step = 10000  # evaluate the agent per training steps
        self.eval_env_class = None  # eval_env = eval_env_class(*eval_env_args)
        self.eval_env_args = None  # eval_env = eval_env_class(*eval_env_args)

        """config for self design"""
        self.stock_num = None

        self.drop = 0.0
        self.leaky_relu = 0.2
        self.nb_heads = 8
        self.hidden = 8
        self.alphas = None

    def init_before_training(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)

        if self.cwd is None:  # set cwd (current working directory) for saving model
            self.cwd = f'./{self.env_name}_{self.agent_class.__name__[5:]}_{self.random_seed}'

        if self.if_remove is None:  # remove or keep the history files
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
        if self.if_remove:
            import shutil
            shutil.rmtree(self.cwd, ignore_errors=True)
            print(f"| Arguments Remove cwd: {self.cwd}")
        else:
            print(f"| Arguments Keep cwd: {self.cwd}")
        os.makedirs(self.cwd, exist_ok=True)
        os.makedirs(f"{self.cwd}/EnvFigs", exist_ok=True)

    def get_if_off_policy(self) -> bool:
        agent_name = self.agent_class.__name__ if self.agent_class else ''
        on_policy_names = ('SARSA', 'VPG', 'A2C', 'A3C', 'TRPO', 'PPO', 'MPO')
        return all([agent_name.find(s) == -1 for s in on_policy_names])

    def print(self):
        from pprint import pprint
        pprint(vars(self))  # prints out args in a neat, readable format

    def save(self):
        file_path = f"{self.cwd}/config.obj"
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

