import torch
from typing import Tuple
from torch import Tensor

from rl_agent.AgentBase import AgentBase
from rl_agent.net import ActorPPO, CriticPPO
from rl_agent.rl_config import RLConfig

"""
PPO algorithm. “Proximal Policy Optimization Algorithms”. John Schulman. et al.. 2017.

net_dims: the middle layer dimension of MLP (MultiLayer Perceptron)
state_dim: the dimension of state (the number of state vector)
action_dim: the dimension of action (or the number of discrete action)
gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
args: the arguments for agent training. `args = Config()`
"""
class AgentPPO(AgentBase):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int,  gpu_id: int = 0, args: RLConfig = RLConfig()):
        self.act_class = getattr(self, "act_class", ActorPPO)
        self.cri_class = getattr(self, "cri_class", CriticPPO)
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        self.if_off_policy = False

        self.ratio_clip = getattr(args, "ratio_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", 0.99)  # could be 0.50~0.99 # GAE for sparse reward
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.001)  # could be 0.00~0.20
        self.lambda_entropy = torch.tensor(self.lambda_entropy, dtype=torch.float32, device=self.device)

        if getattr(args, 'if_use_v_trace', False):
            self.get_advantages = self.get_advantages_vtrace  # get advantage value in reverse time series (V-trace)
        else:
            self.get_advantages = self.get_advantages_origin  # get advantage value using critic network
        self.value_avg = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.value_std = torch.ones(1, dtype=torch.float32, device=self.device)

        self.alphas = args.alphas
        self.stock_num =args.stock_num



    def explore_one_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        states = torch.zeros((horizon_len,self.alphas, self.stock_num), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.float32).to(self.device)
        logprobs = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        state = self.last_state  # shape == (1, state_dim) for a single env.

        get_action = self.act.get_action
        convert = self.act.convert_action_for_env
        for t in range(horizon_len):
            state = state.unsqueeze(0)
            action, logprob = get_action(state)
            states[t] = state[0]

            ary_action = convert(action[0]).detach().cpu().numpy()
            ary_state, reward, done, _ = env.step(ary_action)  # next_state
            state = env.reset() if done else ary_state  # ary_state.shape == (state_dim, )
            # state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            actions[t] = action
            logprobs[t] = logprob
            rewards[t] = reward
            dones[t] = done

        self.last_state = state  # state.shape == (1, state_dim) for a single env.

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, logprobs, rewards, undones


    def update_net(self, buffer) -> Tuple[float, ...]:
        with torch.no_grad():
            states, actions, logprobs, rewards, undones = buffer
            buffer_size = states.shape[0]
            buffer_num = 1

            '''get advantages and reward_sums'''
            bs = 2 ** 10  # set a smaller 'batch_size' to avoiding out of GPU memory.
            values = torch.empty_like(rewards)  # values.shape == (buffer_size, buffer_num)
            for i in range(0, buffer_size, bs):
                values[i:i + bs, 0] = self.cri(states[i:i + bs])

            advantages = self.get_advantages(rewards, undones, values)  # shape == (buffer_size, buffer_num)
            reward_sums = advantages + values  # shape == (buffer_size, buffer_num)
            del rewards, undones, values

            advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-4)

            self.update_avg_std_for_normalization(
                states=states,
                returns=reward_sums.reshape((-1,))
            )
        # assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size, buffer_num)

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0
        sample_len = buffer_size - 1

        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for _ in range(update_times):
            ids = torch.randint(sample_len * buffer_num, size=(self.batch_size,), requires_grad=False)
            ids0 = torch.fmod(ids, sample_len)  # ids % sample_len
            ids1 = torch.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len

            state = states[ids0]
            action = actions[ids0, ids1]
            logprob = logprobs[ids0, ids1]
            advantage = advantages[ids0, ids1]
            reward_sum = reward_sums[ids0, ids1]

            value = self.cri(state)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, reward_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = torch.min(surrogate1, surrogate2).mean()

            obj_actor = obj_surrogate + obj_entropy.mean() * self.lambda_entropy
            self.optimizer_update(self.act_optimizer, -obj_actor)

            obj_critics += obj_critic.item()
            obj_actors += obj_actor.item()
        a_std_log = self.act.action_std_log.mean() if hasattr(self.act, 'action_std_log') else torch.zeros(1)
        return obj_critics / update_times, obj_actors / update_times, a_std_log.item()

    def get_advantages_origin(self, rewards: Tensor, undones: Tensor, values: Tensor) -> Tensor:
        advantages = torch.empty_like(values)  # advantage value

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        next_value = self.cri(self.last_state.unsqueeze(0)).detach()

        advantage = torch.zeros_like(next_value)  # last advantage value by GAE (Generalized Advantage Estimate)
        for t in range(horizon_len - 1, -1, -1):
            next_value = rewards[t] + masks[t] * next_value
            advantages[t] = advantage = next_value - values[t] + masks[t] * self.lambda_gae_adv * advantage
            next_value = values[t]
        return advantages

    def get_advantages_vtrace(self, rewards: Tensor, undones: Tensor, values: Tensor) -> Tensor:
        advantages = torch.empty_like(values)  # advantage value

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        advantage = torch.zeros_like(values[0])  # last advantage value by GAE (Generalized Advantage Estimate)
        for t in range(horizon_len - 1, -1, -1):
            advantages[t] = rewards[t] - values[t] + masks[t] * advantage
            advantage = values[t] + self.lambda_gae_adv * advantages[t]
        return advantages

