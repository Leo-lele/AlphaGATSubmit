import argparse
import os
import time

import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from indicator.datasetstock import Dataset_Stock
from rl_agent.AgentPPO import AgentPPO
from rl_agent.StockEnv import Stock_Env
from rl_agent.evaluator import Evaluator
from rl_agent.replay_buffer import ReplayBuffer
from rl_agent.rl_config import RLConfig


DATASET ="BTC"
DEBUG_MODE = False
ALPHAS = 64
DAY_WINDOWS = 30
MODEL_DIR = "LoadModel/BTC/model.pt"
STOCK_NUM = 18


parser = argparse.ArgumentParser(description='argparse testing')
parser.add_argument('--save_dir','-dir',type=str, default = "./temp",required=False,help="dir to solve the model and logs")
parser.add_argument('--batch_size',type=int, default=1024)
parser.add_argument('--horizon_len',type=int, default=2048)
parser.add_argument('--break_step',type=int, default=200000)

parser.add_argument("--learning_rate", type=float, default=5e-5)

parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--repeat_times", type=float, default=1.0)
parser.add_argument("--drop", type=float, default=0.5)
parser.add_argument("--G", type=int, default=5)
parser.add_argument("--adv", type=float, default=0.95)
parser.add_argument("--entropy", type=float, default=0.01)

args = parser.parse_args()

print("*************************************************")
print('--save_dir', args.save_dir)
print('--batch_size', args.batch_size)
print('--horizon_len', args.horizon_len)
print('--break_step', args.break_step)
print('--learning_rate', args.learning_rate)
print('--gpu_id', args.gpu_id)
print('--repeat_time', args.repeat_times)
print('--drop', args.drop)
print('--G', args.G)
print('--lambda_gae_adv', args.adv)
print('--lambda_entropy', args.entropy)
print("**************************************************")


#####################################################
alphas = ALPHAS
state_dim = alphas * STOCK_NUM
action_dim = alphas



env_args = {
    'env_name': 'StockEnv',  # Apply torque on the free end to swing a pendulum into an upright position
    'max_step': 11111111111,  # the max step number of an episode.
    'state_dim': state_dim,  # the x-y coordinates of the pendulum's free end and its angular velocity.
    'action_dim': action_dim,  # the torque applied to free end of the pendulum
    'if_discrete': False  # continuous action space, symbols → direction, value → force
}
config = RLConfig(agent_class=AgentPPO, env_class=Stock_Env, env_args=env_args)
config.cwd = args.save_dir
config.batch_size = args.batch_size
config.horizon_len = args.horizon_len
config.break_step = args.break_step
config.learning_rate = args.learning_rate
config.gpu_id = args.gpu_id
config.repeat_times = args.repeat_times
config.drop = args.drop
config.lambda_gae_adv = args.adv
config.lambda_entropy = args.entropy
config.init_before_training()
config.stock_num = STOCK_NUM
config.alphas = ALPHAS

if DEBUG_MODE:
    config.batch_size = 4
    config.horizon_len = 8
    config.break_step = 16
    config.eval_per_step = 8


####设置参数
torch.set_grad_enabled(False)

device = f"cuda:{config.gpu_id}"

train_dataset = Dataset_Stock(dataset = DATASET,day_windows=DAY_WINDOWS,
                              flag="train", device=device)

valid_dataset = Dataset_Stock(dataset = DATASET,day_windows=DAY_WINDOWS,
                              flag="valid", device=device)

alpha_model = torch.load(MODEL_DIR, map_location=device)

stock_env_args = {
    "stock_num": STOCK_NUM,
    "state_dim": state_dim,
    "G": args.G,
    "cwd": config.cwd
}

train_env = Stock_Env(dataset=train_dataset, model=alpha_model, **stock_env_args,
                      mode='train')



#########eval_env setting
eval_train_env = Stock_Env(dataset=train_dataset, model=alpha_model, **stock_env_args,
                      mode='train', is_eval=True)
eval_valid_env = Stock_Env(dataset=valid_dataset, model=alpha_model, **stock_env_args,
                      mode='eval', is_eval=True)

eval_env_list = [eval_train_env, eval_valid_env]


agent = config.agent_class( state_dim=config.state_dim,
                         action_dim = config.action_dim, gpu_id = config.gpu_id, args=config)
agent.save_or_load_agent(config.cwd, if_save=False)
state = train_env.reset()
agent.last_state = state.detach()

'''init buffer'''
if config.if_off_policy:
    buffer = ReplayBuffer(
        gpu_id=config.gpu_id,
        num_seqs=config.num_envs,
        max_size=config.buffer_size,
        state_dim=config.state_dim,
        action_dim=1 if config.if_discrete else config.action_dim,
        if_use_per=config.if_use_per,
        args=config,
    )
    buffer_items = agent.explore_env(train_env, config.horizon_len * config.eval_times, if_random=True)
    buffer.update(buffer_items)  # warm up for ReplayBuffer
else:
    buffer = []

evaluator = Evaluator(cwd=config.cwd, envs_list=eval_env_list, args=config, if_tensorboard=True)

'''train loop'''
cwd = config.cwd
break_step = config.break_step
horizon_len = config.horizon_len
if_off_policy = config.if_off_policy
if_save_buffer = config.if_save_buffer
config.save()

del config

if_train = True
while if_train:
    buffer_items = agent.explore_env(train_env, horizon_len)

    exp_r = buffer_items[3].mean().item()
    if if_off_policy:
        buffer.update(buffer_items)
    else:
        buffer[:] = buffer_items

    torch.set_grad_enabled(True)
    logging_tuple = agent.update_net(buffer)
    torch.set_grad_enabled(False)

    evaluator.evaluate_and_save(actor=agent.act, steps=horizon_len, exp_r=exp_r, logging_tuple=logging_tuple)
    if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))

print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')
evaluator.save_Figs_end()
agent.save_or_load_agent(cwd, if_save=True)

