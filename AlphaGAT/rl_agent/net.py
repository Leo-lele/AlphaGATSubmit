import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal

from rl_agent.GAT import GAT
from rl_agent.rl_config import RLConfig

"""Actor (policy network)"""

class ActorBase(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, alphas):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim, *dims, action_dim])
        self.explore_noise_std = None  # standard deviation of exploration action noise
        self.ActionDist = torch.distributions.normal.Normal

        self.state_avg = nn.Parameter(torch.zeros((alphas, 1)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((alphas, 1)), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std



class ActorPPO(ActorBase):
    def __init__(self,  state_dim: int, action_dim: int, args:RLConfig()):

        super().__init__(state_dim=state_dim, action_dim=action_dim, alphas=args.alphas)

        self.net = GAT(args.stock_num, args.hidden, args.drop, args.leaky_relu, args.nb_heads)

        self.action_std_log = nn.Parameter(torch.zeros((1, action_dim)), requires_grad=True)  # trainable parameter

    def forward(self, state: Tensor) -> Tensor:
        state = self.state_norm(state)
        return self.net(state)


    def get_action(self, state: Tensor) -> (Tensor, Tensor):  # for exploration
        state = self.state_norm(state)

        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = self.ActionDist(action_avg, action_std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        return action, logprob

    def get_logprob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        state = self.state_norm(state)
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = self.ActionDist(action_avg, action_std)
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        return action


class CriticBase(nn.Module):  # todo state_norm, value_norm
    def __init__(self, state_dim: int, action_dim: int, alphas):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim + action_dim, *dims, 1])

        self.state_avg = nn.Parameter(torch.zeros((alphas, 1)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((alphas, 1)), requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std  # todo state_norm

    def value_re_norm(self, value: Tensor) -> Tensor:
        return value * self.value_std + self.value_avg  # todo value_norm



class CriticPPO(CriticBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int, args:RLConfig()):
        super().__init__(state_dim=state_dim, action_dim=action_dim, alphas=args.alphas)
        self.net = GAT(args.stock_num, args.hidden, args.drop, args.leaky_relu, args.nb_heads)
        self.mlp = nn.Linear(args.alphas, 1)

    def forward(self, state: Tensor) -> Tensor:


        state = self.state_norm(state)

        value = self.net(state)
        result2 = self.mlp(value).squeeze(-1)
        result = self.value_re_norm(result2)
        return result


###build MLP (MultiLayer Perceptron) with dropout layer
def build_mlp(dims: [int], activation: nn = None, if_raw_out: bool = True, dropout_prob: float = 0.5) -> nn.Sequential:
    if activation is None:
        activation = nn.LeakyReLU
    net_list = []
    for i in range(len(dims) - 2):
        # Add dropout layer before activation for hidden layers
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.Dropout(dropout_prob), activation()])
    # For the last layer, don't add activation and dropout if if_raw_out is True
    if if_raw_out:
        net_list.extend([nn.Linear(dims[-2], dims[-1])])
    else:
        net_list.extend([nn.Linear(dims[-2], dims[-1]), nn.Dropout(dropout_prob), activation()])
    return nn.Sequential(*net_list)

def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


def mlp_init_with_orthogonal(mlp, std=1.0, bias_const=1e-6):
    for i, layer in enumerate(mlp):
        if isinstance(layer, nn.Linear):
            layer_init_with_orthogonal(layer, std=0.01, bias_const=0.0)

