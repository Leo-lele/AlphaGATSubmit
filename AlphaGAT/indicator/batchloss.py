import torch.nn as nn
import torch
from indicator.config import Config
import torch.nn.functional as F

class Batch_Loss(nn.Module):
    def __init__(self, config:Config):
        super(Batch_Loss, self).__init__()
        self.EPS = 1e-6
        self.use_cov = config.use_cov
        self.factor_cov = config.factor_cov


    ####The correlation is calculated using IC values, with the objective of
    # maximizing the absolute value of the correlation.

    def forward(self, preds, targets):
        B, N, S = preds.shape
        ##IC_Loss = loss 1  [batch, indicators]
        ##CM loss [batch, num_alphas, num_alphas]
        IC_Loss = self.loss_IC(preds, targets)

        IC_Loss = IC_Loss.mean()

        CM_loss = self.loss_covariance_matrix(preds)

        CM_loss = self.factor_cov * torch.sum(CM_loss) / (N **2)
        if self.use_cov:
            loss = IC_Loss - self.factor_cov * CM_loss
        else:
            loss = IC_Loss.mean()

        return loss, IC_Loss, CM_loss


    def loss_covariance_matrix(self, preds):
        #####preds.shape == (batch, num_alphas, stock_num)
        B, N, S = preds.shape
        pred_centered = preds - preds.mean(dim=2, keepdim=True)
        cov_matrix = torch.matmul(pred_centered, pred_centered.transpose(1, 2)) / (S - 1)
        loss = cov_matrix ** 2
        ####loss  [batch, num_alphas]
        return loss


    def loss_IC(self, preds, targets):
        #####preds.shape ==(batch, num_indicator, stock_num)
        #####targets.shape == (batch, 1, stock_num)

        mean_preds = preds.mean(dim=-1, keepdim=True)
        mean_targets = targets.mean(dim=-1, keepdim=True)

        var_preds = torch.pow((preds - mean_preds), 2).sum(dim=-1)
        var_targets = torch.pow((targets - mean_targets), 2).sum(dim=-1)

        cov = ((preds - mean_preds) * (targets - mean_targets)).sum(dim=-1)

        ####loss 1  [batch, indicators, indicators]
        loss = cov / torch.sqrt((var_preds * var_targets) + self.EPS)

        return loss





