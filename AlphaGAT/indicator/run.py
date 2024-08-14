import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import sys


import sys
sys.path.append('调用的包的路径')
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
####  从命令行读取参数
from indicator.config import Config
from indicator.trainer import Trainer

parser = argparse.ArgumentParser(description='argparse testing')
parser.add_argument('--dataset', type=str, default='BTC')
parser.add_argument('--save_dir','-dir',type=str, default = "./temp",required=True,help="dir to solve the model and logs")
parser.add_argument('--batch_size',type=int, default=256)
parser.add_argument('--day_windows',type=int, default=30)
parser.add_argument('--epochs',type=int, default=200)
parser.add_argument("--learning_rate", type=float, default=5e-2)
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--drop", type=float, default=0.5)
parser.add_argument("--alphas", type=int, default=64)
parser.add_argument("--model", type=str, default='IMM')
parser.add_argument('--use_cov', action='store_true', default=False)
parser.add_argument('--use_attention', action='store_true', default=False)
parser.add_argument('--use_cov1d', action='store_true', default=False)
parser.add_argument('--factor_cov', type=float, default=1e-5)
parser.add_argument("--d_model", type=int, default=64)
args = parser.parse_args()
print("*************************************************")
print('--dataset', args.dataset)
print('--save_dir', args.save_dir)
print('--batch_size', args.batch_size)
print('--day_windows', args.day_windows)
print('--epochs', args.epochs)
print('--learning_rate', args.learning_rate)
print('--gpu_id', args.gpu_id)
print('--drop', args.drop)
print('--alphas', args.alphas)
print('--model', args.model)
print('--use_cov', args.use_cov)
print('--factor_cov', args.factor_cov)
print('--use_attention', args.use_attention)
print('--use_cov1d', args.use_cov1d)
print('--d_model', args.d_model)
print("**************************************************")
####

####   设置参数并且保存参数
config = Config(args)
config.init_before_training()
config.save()

# """构建网络模型比能切"""
# stock_series  =torch.rand(2, 30, 29, 7);
#
# ars  =Config(args)
# model = ModernTCN(ars)
# pred_series = model(stock_series)
# print(pred_series.shape)




trainer = Trainer(config)
trainer.run()

# def main():
#     if DEBUG_MODE:
#         _set_debug_mode()
#
#     create_logger(**logger_params)
#     _print_config()
#
#     trainer = Trainer(env_params=env_params,
#                       model_params=model_params,
#                       optimizer_params=optimizer_params,
#                       trainer_params=trainer_params)
#
#     copy_all_src(trainer.result_folder)
#
#     trainer.run()
#
#
# if __name__ == "__main__":
#     main()
