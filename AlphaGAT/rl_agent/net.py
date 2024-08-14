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
    def __init__(self, dims: [int], state_dim: int, action_dim: int, args:RLConfig()):

        super().__init__(state_dim=state_dim, action_dim=action_dim, alphas=args.alphas)
        ##self.net = GAT(args.stock_num, args.hidden, args.drop, args.leaky_relu, args.nb_heads)

        self.net = GAT(args.stock_num, args.hidden, args.drop, args.leaky_relu, args.nb_heads)

        # self.net = build_mlp(dims=[args.stock_num, *dims, 1], dropout_prob=args.drop)
        # layer_init_with_orthogonal(self.net[-1], std=0.1)

        self.action_std_log = nn.Parameter(torch.zeros((1, action_dim)), requires_grad=True)  # trainable parameter

    def forward(self, state: Tensor) -> Tensor:
        state = self.state_norm(state)
        return self.net(state)
        # return self.net(state).squeeze(-1)
        #return self.net(state).tanh() # action.tanh()

    def get_action(self, state: Tensor) -> (Tensor, Tensor):  # for exploration
        state = self.state_norm(state)

        action_avg = self.net(state)
        # action_avg = self.net(state).squeeze(-1)
        action_std = self.action_std_log.exp()

        dist = self.ActionDist(action_avg, action_std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        return action, logprob

    def get_logprob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        state = self.state_norm(state)
        action_avg = self.net(state)
        # action_avg = self.net(state).squeeze(-1)
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
        #layer_init_with_orthogonal(self.net[-1], std=0.5)
        # self.net = build_mlp(dims=[args.stock_num, *dims, 1], dropout_prob=args.drop)


    def forward(self, state: Tensor) -> Tensor:


        state = self.state_norm(state)
        # result2 = self.net(state).squeeze(-1)
        # result2 = self.mlp(result2).squeeze(-1)
        value = self.net(state)
        result2 = self.mlp(value).squeeze(-1)

        # result = state.permute(0, 2, 1) * value.unsqueeze(1)
        # temp = torch.sum(result, dim=-1)
        # temp = torch.mean(temp, dim=-1)
        #
        # result = torch.mean(result, dim=(1,2))
        result = self.value_re_norm(result2)
        return result


# def build_mlp(dims: [int], activation: nn = None, if_raw_out: bool = True) -> nn.Sequential:
#     """
#     build MLP (MultiLayer Perceptron)
#
#     dims: the middle dimension, `dims[-1]` is the output dimension of this network
#     activation: the activation function
#     if_remove_out_layer: if remove the activation function of the output layer.
#     """
#     if activation is None:
#         activation = nn.ReLU
#     net_list = []
#     for i in range(len(dims) - 1):
#         net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
#     if if_raw_out:
#         del net_list[-1]  # delete the activation function of the output layer to keep raw output
#     return nn.Sequential(*net_list)
#
#
# def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
#     torch.nn.init.orthogonal_(layer.weight, std)
#     torch.nn.init.constant_(layer.bias, bias_const)


def build_mlp(dims: [int], activation: nn = None, if_raw_out: bool = True, dropout_prob: float = 0.5) -> nn.Sequential:
    """
    build MLP (MultiLayer Perceptron) with dropout layer

    dims: the middle dimension, `dims[-1]` is the output dimension of this network
    activation: the activation function
    if_remove_out_layer: if remove the activation function of the output layer.
    dropout_prob: dropout probability for dropout layer
    """
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



# class NnReshape(nn.Module):
#     def __init__(self, *args):
#         super().__init__()
#         self.args = args
#
#     def forward(self, x):
#         return x.view((x.size(0),) + self.args)
#
#
# class DenseNet(nn.Module):  # plan to hyper-param: layer_number
#     def __init__(self, lay_dim):
#         super().__init__()
#         self.dense1 = nn.Sequential(nn.Linear(lay_dim * 1, lay_dim * 1), nn.Hardswish())
#         self.dense2 = nn.Sequential(nn.Linear(lay_dim * 2, lay_dim * 2), nn.Hardswish())
#         self.inp_dim = lay_dim
#         self.out_dim = lay_dim * 4
#
#     def forward(self, x1):  # x1.shape==(-1, lay_dim*1)
#         x2 = torch.cat((x1, self.dense1(x1)), dim=1)
#         return torch.cat(
#             (x2, self.dense2(x2)), dim=1
#         )  # x3  # x2.shape==(-1, lay_dim*4)
#
#
# class ConvNet(nn.Module):  # pixel-level state encoder
#     def __init__(self, inp_dim, out_dim, image_size=224):
#         super().__init__()
#         if image_size == 224:
#             self.net = nn.Sequential(  # size==(batch_size, inp_dim, 224, 224)
#                 nn.Conv2d(inp_dim, 32, (5, 5), stride=(2, 2), bias=False),
#                 nn.ReLU(inplace=True),  # size=110
#                 nn.Conv2d(32, 48, (3, 3), stride=(2, 2)),
#                 nn.ReLU(inplace=True),  # size=54
#                 nn.Conv2d(48, 64, (3, 3), stride=(2, 2)),
#                 nn.ReLU(inplace=True),  # size=26
#                 nn.Conv2d(64, 96, (3, 3), stride=(2, 2)),
#                 nn.ReLU(inplace=True),  # size=12
#                 nn.Conv2d(96, 128, (3, 3), stride=(2, 2)),
#                 nn.ReLU(inplace=True),  # size=5
#                 nn.Conv2d(128, 192, (5, 5), stride=(1, 1)),
#                 nn.ReLU(inplace=True),  # size=1
#                 NnReshape(-1),  # size (batch_size, 1024, 1, 1) ==> (batch_size, 1024)
#                 nn.Linear(192, out_dim),  # size==(batch_size, out_dim)
#             )
#         elif image_size == 112:
#             self.net = nn.Sequential(  # size==(batch_size, inp_dim, 112, 112)
#                 nn.Conv2d(inp_dim, 32, (5, 5), stride=(2, 2), bias=False),
#                 nn.ReLU(inplace=True),  # size=54
#                 nn.Conv2d(32, 48, (3, 3), stride=(2, 2)),
#                 nn.ReLU(inplace=True),  # size=26
#                 nn.Conv2d(48, 64, (3, 3), stride=(2, 2)),
#                 nn.ReLU(inplace=True),  # size=12
#                 nn.Conv2d(64, 96, (3, 3), stride=(2, 2)),
#                 nn.ReLU(inplace=True),  # size=5
#                 nn.Conv2d(96, 128, (5, 5), stride=(1, 1)),
#                 nn.ReLU(inplace=True),  # size=1
#                 NnReshape(-1),  # size (batch_size, 1024, 1, 1) ==> (batch_size, 1024)
#                 nn.Linear(128, out_dim),  # size==(batch_size, out_dim)
#             )
#         else:
#             assert image_size in {224, 112}
#
#     def forward(self, x):
#         # assert x.shape == (batch_size, inp_dim, image_size, image_size)
#         x = x.permute(0, 3, 1, 2)
#         x = x / 128.0 - 1.0
#         return self.net(x)
#
#     @staticmethod
#     def check():
#         inp_dim = 3
#         out_dim = 32
#         batch_size = 2
#         image_size = [224, 112][1]
#         # from elegantrl.net import Conv2dNet
#         net = ConvNet(inp_dim, out_dim, image_size)
#
#         image = torch.ones((batch_size, image_size, image_size, inp_dim), dtype=torch.uint8) * 255
#         print(image.shape)
#         output = net(image)
#         print(output.shape)

# dims = [16,256,256,1]
# mlp = build_mlp(dims, activation=nn.ReLU, if_raw_out=True, dropout_prob=0.5)
# in_data = torch.randn((16,16))
# out = mlp(in_data)
# print(out.shape)
# print(out)