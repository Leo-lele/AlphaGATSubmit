import argparse
import os
import time

import numpy as np
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

root_path = os.path.abspath('~/Desktop/lsc/code/Alphas_Stock/v1/indicator')
sys.path.append(root_path)
root_path = os.path.abspath('~/Desktop/lsc/code/Alphas_Stock/v1')
sys.path.append(root_path)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)


from indicator.datasetstock import Dataset_Stock
from rl_agent.AgentPPO import AgentPPO
from rl_agent.StockEnv import Stock_Env
from rl_agent.evaluator import Evaluator
from rl_agent.replay_buffer import ReplayBuffer
from rl_agent.rl_config import RLConfig


G = 7
DEBUG_MODE = False
ALPHAS = 64
STOCK_DATA_DIR = "../../data/DIJA"
DAY_WINDOWS = 30
MODEL_DIR = "LoadModel/DIJA/model.pt"
STOCK_NUM = 29
alphas = ALPHAS
state_dim = alphas * STOCK_NUM
action_dim = alphas
torch.set_grad_enabled(False)

device = f"cuda:0"

train_dataset = Dataset_Stock(data_dir=STOCK_DATA_DIR,day_windows=DAY_WINDOWS,
                              flag="train", device=device)

valid_dataset = Dataset_Stock(data_dir=STOCK_DATA_DIR,day_windows=DAY_WINDOWS,
                              flag="valid", device=device)
test_dataset = Dataset_Stock(data_dir=STOCK_DATA_DIR,day_windows=DAY_WINDOWS,
                              flag="test", device=device)

alpha_model = torch.load(MODEL_DIR, map_location=device)

stock_env_args = {
    "stock_num": STOCK_NUM,
    "state_dim": state_dim,
    "G": G,
    "cwd": "equal"
}




#########eval_env setting
eval_train_env = Stock_Env(dataset=train_dataset, model=alpha_model, **stock_env_args,
                      mode='train', is_eval=True)
eval_valid_env = Stock_Env(dataset=valid_dataset, model=alpha_model, **stock_env_args,
                      mode='eval', is_eval=True)
eval_test_env = Stock_Env(dataset=test_dataset, model=alpha_model, **stock_env_args,
                      mode='test', is_eval=True)

eval_env_list = [eval_train_env, eval_valid_env, eval_test_env]



print("eqaul_weight********")
for env in eval_env_list:
    state = env.reset()
    steps = None
    returns = 0.0  # sum of rewards in an episode
    max_step = 999999

    with torch.no_grad():
        for steps in range(max_step):

            action =np.array( [1.0 / alphas] * alphas)
            state, reward, done, _ = env.step(action)
            returns += reward

            if done:
                break

env  = eval_test_env
print("best  weight********")
for i in range(ALPHAS):
    state = env.reset()
    steps = None
    returns = 0.0  # sum of rewards in an episode
    max_step = 999999

    with torch.no_grad():
        for steps in range(max_step):

            action =np.zeros((ALPHAS))
            action[i] = 1.0
            state, reward, done, _ = env.step(action)
            returns += reward

            if done:
                break

print("random weight ********")
state = env.reset()
steps = None
returns = 0.0  # sum of rewards in an episode
max_step = 999999

with torch.no_grad():
    for steps in range(max_step):

        action =np.random.rand(ALPHAS)
        action /= np.sum(action)
        state, reward, done, _ = env.step(action)
        returns += reward

        if done:
            break