import torch
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
import numpy





import torch
from torch.distributions.uniform import Uniform

import torch
import torch.distributions as dist

import torch
from torch.distributions.uniform import Uniform

import torch

import torch
import torch.distributions as dist

import torch
import torch.distributions as dist

import torch
import torch.distributions as dist

import torch
from torch.distributions.uniform import Uniform

import torch
import torch.distributions as dist

import torch

import torch
from torch.distributions.uniform import Uniform

def forward_diffusion_sample_chebyshev(x_0, t, constant_dict, config):
    """ 
    Takes an image and a timestep as input and 
    returns the version of it with Chebyshev noise.
    """
    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = constant_dict['sqrt_alphas_cumprod'], constant_dict['sqrt_one_minus_alphas_cumprod']

    # Create Chebyshev noise
    # For simplicity, we're using first-order Chebyshev polynomial
    uniform_dist = Uniform(-1, 1)
    noise = uniform_dist.sample(x_0.shape)
    chebyshev_noise = torch.cos(torch.acos(noise))

    device = config.model.device

    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape, config
    )

    # Apply Chebyshev noise
    x = sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * (x_0.to(device) * chebyshev_noise.to(device))

    x = x.to(device)
    noise = noise.to(device)  # Returning the original noise

    return x, noise

















def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

