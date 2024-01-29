import torch
import os
import torch.nn as nn
import numpy as np
from forward_process import *


def get_loss(model, x_0, t, config):
    x_0 = x_0.to(config.model.device)
    betas = np.linspace(config.model.beta_start, config.model.beta_end, config.model.trajectory_steps, dtype=np.float64)
    b = torch.tensor(betas).type(torch.float).to(config.model.device)
    e = torch.randn_like(x_0, device = x_0.device)
    at = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    sqrt_at = at.sqrt()
    sqrt_complement_at = (1 - at).sqrt()
    x = sqrt_at * x_0 + sqrt_complement_at * e
    output = model(x, t.float())

    # Original MSE loss between noise e and model output
    mse_loss = (e - output).square().sum(dim=(1, 2, 3))

    # L2 Regularization term for smoothness in the model's output
    l2_lambda = 0.01  # Regularization strength, can be adjusted
    l2_reg = output.square().sum()

    # Combined loss
    combined_loss = mse_loss + l2_lambda * l2_reg

    # Returning the mean of the combined loss
    return combined_loss.mean(dim=0)



