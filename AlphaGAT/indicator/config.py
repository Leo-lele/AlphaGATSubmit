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
        assert self.dataset in ['BTC', 'CSI100', 'DIJA', 'HSI']

        '''定义需要使用到的指标，可以自己添加或者选择'''

        if self.dataset == 'BTC':
            self.INDICATORS = [
                "close",
                "open",
                "high",
                "low",
                'volume'
            ]
            self.stock_num = 18
        elif self.dataset == 'CSI100':
            self.INDICATORS = [
                "close",
                "open",
                "high",
                "low",
                'vwap',
                'turn',
                'chg',
                'swing',
                'volume'
            ]
            self.stock_num = 98

        elif self.dataset == 'HSI':
            self.INDICATORS = [
                "close",
                "open",
                "high",
                "low",
                'vwap',
                'turn',
                'chg',
                'swing',
                'volume'
            ]
            self.stock_num = 56

        elif self.dataset == 'DIJA':
            self.INDICATORS = [
                "close",
                "open",
                "high",
                "low",
                'turn',
                'chg',
                'volume'
            ]
            self.stock_num = 29
        else:
            raise ValueError("This dataset is not supported ")

        self.indicator_num = len(self.INDICATORS)
        # self.stock_num = 18
        # self.alphas = 512

        '''Arguments for network'''
        '''hyper-parameters for TCN'''
        self.tcn_kernel_size = 3
        self.tcn_levels = 4
        self.tcn_hidden_dim = 512
        # '''hyper-parameters for LSTM'''
        # self.lstm_input_dim = 128
        # self.lstm_hidden_size = 128

        '''hyper-parameters for Transformer'''
        self.embedding_dim = 512
        self.sqrt_embedding_dim = 512**(1/2)
        self.qkv_dim = 64
        self.head_num = 8
        self.logit_clipping = 10
        self.ff_hidden_dim = 512

        '''hyper-parameters for Polcicy Net'''
        self.lstm_layer = 1
        self.encoder_layer_num = 3

        self.soft_temp = 2

        '''hyper-parameters for ITransformer'''
        self.e_layer = 3

        '''hyper-parameters for ModernTCN'''
        self.pred_len = 1
        self.mod_hidden_dim=512
        self.P = 8
        self.S = 4
        self.mod_kernel_size = 13
        self.mod_layers = 2


        '''hyper-parameters for TimeMix'''
        self.seq_len  =self.day_windows
        self.pred_len = 1

        ##向下采样的窗口
        self.down_sampling_window = 2
        ###projection 时每个通道是否是独立的  1 默认时独立的
        self.channel_independence = 1

        ###pdm_blocks  的层数  原文中为 2 3 4 5
        self.e_layers  = 3
        ##d_model=32 128 16
        self.d_model = args.d_model

        ###d_ff  32 64 128  256
        self.d_ff = 64

        self.dropout = self.drop

        ####向下采样的方式不同  moving_avg   dft_decomp
        self.decomp_method = "moving_avg"

        ####dft_decomp
        self.top_k = 5

        #####moving_avg  kernel_size  原文25
        self.moving_avg = 7
        ###down_sampling_layers   1 , 3  season  trend  num of layers
        self.down_sampling_layers = 3
        ###max  avg  conv
        self.down_sampling_method = 'avg'

        ###输入通道的独立性  可以看成股票的维度
        self.enc_in = self.indicator_num

        ##timeF' 'time features encoding, options:[timeF, fixed, learned]'
        self.embed = 'timeF'

        #'freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h'
        self.freq = 'h'
        ##whether to use normalize; True 1 False 0
        self.use_norm = 1
        ###'--c_out', type=int, default=7, help='output size'
        self.c_out  = self.indicator_num

        self.attention_dim = 256

        ####trend和seasonal融合时是否采取cov1d还是简单的相加.
        ####是否使用attention 来提取不同stcok之间的相关性
        self.use_cov1d = args.use_cov1d
        self.use_attention = args.use_attention





        '''training'''
        self.clip_grad_norm = 10
        self.log_file = f"{self.cwd}/training.log"
        self.use_cov = args.use_cov
        self.factor_cov = args.factor_cov

        '''model save'''
        self.if_over_write = True
        self.save_gap = 40


    def init_before_training(self):

        ###设置随机数的种子
        # seed = self.random_seed
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # ra
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

