import os
import pickle
from random import random

import numpy as np
import torch

class Config:
    def  __init__(self, args):
        self.cwd = args.save_dir
        self.batch_size = args.batch_size
        self.day_windows = args.day_windows
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.drop = args.drop
        self.alphas = args.alphas
        self.device = torch.device(f"cuda:{args.gpu_id}" if (torch.cuda.is_available() and (args.gpu_id >= 0)) else "cpu")
        self.model = args.model
        self.if_remove = True
        self.dataset = args.dataset
        self.random_seed = 0
        '''Arguments for data'''
        self.dataset == 'BTC'
        self.INDICATORS = [
                "close",
                "open",
                "high",
                "low",
                'vwap',
                'turn',
                'chg',
                'volume'
            ]

        self.indicator_num = len(self.INDICATORS)

        '''Arguments for network'''
        self.seq_len = self.day_windows
        self.pred_len = 1
        self.down_sampling_window = 2
        self.channel_independence = 1

        self.e_layers  = 3
        ##d_model=32 128 16
        self.d_model = args.d_model

        ###d_ff  32 64 128  256
        self.d_ff = 64

        self.dropout = self.drop
        self.decomp_method = "moving_avg"
        #####moving_avg  kernel_size
        self.moving_avg = 7
        ###down_sampling_layers   1 , 3  season  trend  num of layers
        self.down_sampling_layers = 3
        self.down_sampling_method = 'avg'
        self.enc_in = self.indicator_num
        self.use_norm = 1
        self.c_out  = self.indicator_num
        self.attention_dim = 256
        self.use_cov1d = args.use_cov1d
        self.use_attention = args.use_attention
        self.head_num = 4
        '''training'''
        self.clip_grad_norm = 10
        self.log_file = f"{self.cwd}/training.log"
        self.use_cov = args.use_cov
        self.factor_cov = args.factor_cov

        '''model save'''
        self.if_over_write = True
        self.save_gap = 40


    def init_before_training(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_default_dtype(torch.float32)

        '''set cwd (current working directory) for saving model'''
        if self.cwd is None:  # set cwd (current working directory) for saving model
            self.cwd = f'./{self.day_windows}_{self.learning_rate.__name__[5:]}_{self.random_seed}'

        '''remove history'''
        if self.if_remove is None:
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
        if self.if_remove:
            import shutil
            shutil.rmtree(self.cwd, ignore_errors=True)
            print(f"| Arguments Remove cwd: {self.cwd}")
        else:
            print(f"| Arguments Keep cwd: {self.cwd}")
        os.makedirs(self.cwd, exist_ok=True)

    def print(self):
        from pprint import pprint
        pprint(vars(self))  # prints out args in a neat, readable format

    def save(self):
        file_path = f"{self.cwd}/config.obj"
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

