import torch
import torch.optim as optim
import torch.distributions as D
import torch.utils.data as data
import torch.distributions.constraints as C
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = list(shape)

    def forward(self, x):
        return x.view([-1] + self.shape)


activation_map = {
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'softmax': nn.Softmax,
    'leakyrelu': nn.LeakyReLU,
    'none': None
}
layer_map = {
    "conv2d": nn.Conv2d,
    "convtranspose2d": nn.ConvTranspose2d,
    "linear": nn.Linear
}
reshape_map = {
    "reshape": Reshape
}


def to_torch(numpy_in, device="cuda", check=False):
    if check and isinstance(numpy_in, torch.Tensor):
        return numpy_in.to(device)
    else:
        return torch.from_numpy(numpy_in).to(device)


def torch_clip(torch_in, low, high):
    return torch.where(torch.where(torch_in >= low, torch_in, low) <= high, torch_in, high)


def to_numpy(torch_in, check=False):
    if check and isinstance(torch_in, np.ndarray):
        return torch_in
    return torch_in.detach().cpu().numpy()


def split_dim(torch_in, dim, new_shape):
    sh = list(torch_in.shape)
    assert dim < len(sh)
    assert sh[dim] == np.prod(new_shape)
    new_shape = sh[:dim] + list(new_shape) + sh[dim + 1:]
    return torch_in.view(new_shape)


# All torch tensors, same size, P(targ | mu, sigma)
def log_gaussian_prob(mu_obs, sigma_obs, targ_obs):
    assert mu_obs.shape == sigma_obs.shape == targ_obs.shape, "%s, %s, %s" % \
                                                              (mu_obs.shape, sigma_obs.shape, targ_obs.shape)
    # assume last dimension is N
    N = mu_obs.shape[-1]

    det = (sigma_obs ** 2).prod(dim=-1)  # should be (batch, num_models) or just (batch,)
    a = torch.log((2 * np.pi)**N * det)
    b = ((targ_obs - mu_obs)**2 / sigma_obs**2).sum(-1)  # (batch, nm, 3) -> (batch, nm) or without num_models

    assert a.shape == b.shape

    # cov determinant term + scaled squared error
    return - 0.5 * (a + b).mean()  # mean over batch and num models


# All torch tensors
def kl_regularization(latent_mu, latent_log_sigma, mean_p=0., sigma_p=1.):
    var_q = (latent_log_sigma.exp()) ** 2
    var_p = sigma_p ** 2
    sigma_p = torch.tensor(sigma_p, device=latent_mu.device)
    kl = ((latent_mu - mean_p) ** 2 + var_q) / (2. * var_p) + (torch.log(sigma_p) - latent_log_sigma) - 0.5
    return kl.mean()


# history is N x hist_len x dim, obs is N x dim
# prepends obs to history along second to last dimension
def advance_history(history, obs):
    # print(history.shape)
    if history.shape[-1] == 0:
        return history

    longer = torch.cat([obs.unsqueeze(-2), history], dim=-2)
    return longer[:, :-1]  # delete the last element
