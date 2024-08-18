import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import random
import sys
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
parser.add_argument('--use_cov', action='store_true', default=True)
parser.add_argument('--use_attention', action='store_true', default=True)
parser.add_argument('--use_cov1d', action='store_true', default=True)
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

####   Set and save parameters.
config = Config(args)
config.init_before_training()
config.save()

trainer = Trainer(config)
trainer.run()

