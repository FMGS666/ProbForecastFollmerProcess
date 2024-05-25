import torch
import random

import numpy as np

from tqdm.notebook import tqdm


# defining function for ensuring reproducible results
def ensure_reproducibility(SEED):
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = True  
    torch.manual_seed(SEED)  
    np.random.seed(SEED)  
    random.seed(SEED)

# defining interpolant configurations
interpolant = {
    "alpha": lambda s: 1 - s,
    "beta": lambda s: s**2, # s
    "sigma": lambda s: 1 - s
}

# defining velocity configurations
velocity = {
    "alpha_dot": lambda s: -1,
    "beta_dot": lambda s: 2*s, # 1
    "sigma_dot": lambda s: -1
}

# defining function for computing gradient of gaussian log pdf
def grad_log_pdf(x, pdf, device):
    x_ = x.clone().detach().requires_grad_(True)
    y = pdf.log_prob(x_.to(device))
    y.retain_grad()
    y.backward()
    return x_.grad

# defining custom follmer process
def g_follmer(s):
    sigma = interpolant["sigma"](s)
    beta = interpolant["beta"](s)
    beta_dot = velocity["beta_dot"](s)
    sigma_dot = velocity["sigma_dot"](s)
    g = 2*s*sigma*(beta_dot*sigma/beta - sigma_dot) - sigma**2
    g = torch.abs(g)
    g = torch.sqrt(g)
    return g

# defining function for creating the context
def pair_lagged_observations(observation_store, lag):
    num_observations = observation_store.shape[0]
    indexes = torch.arange(num_observations - lag)
    current_observation = observation_store[indexes]
    next_observation = observation_store[indexes + 1]
    return current_observation, next_observation

# defining function for creating the random context specified in the paper
def pair_lagged_observations_with_random_context(tensor_video):
    num_frames = tensor_video.shape[0]
    indexes = torch.arange(num_frames - 1)
    next_indexes = indexes + 1
    random_context_indexes = [torch.randint(high = x.item(), size = (1, )) for x in next_indexes]
    target_frame = tensor_video[next_indexes]
    past_frame = tensor_video[indexes]
    random_context_frame = tensor_video[random_context_indexes]
    return (target_frame, past_frame, random_context_frame)

# defining function for creating a lagged dataset with random context from a tensor holding a video
def get_video_dataset_with_random_context(dataset_constructor, video_tensor, device):
    target_frame, past_frame, random_context_frame = pair_lagged_observations_with_random_context(video_tensor)
    video_dataset_with_random_context = dataset_constructor(past_frame, target_frame, random_context_frame, device)
    return video_dataset_with_random_context

# function for plotting densities heatmap
def plot_density(x, y, fig, axes, title = "", bins = 200, ax_bound = None):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    extent = [-ax_bound, ax_bound]*2 if ax_bound is not None else [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    axes.imshow(heatmap.T, extent=extent, origin='lower', cmap = "grey")
    axes.set_title(title)

# function for simulating multimodal jump diffusion process
def simulate_jump_diffusion(simulation_conf):
    # settings of the simulation
    delta_t = simulation_conf["delta_t"]
    observation_interval = simulation_conf["observation_interval"]

    # defining number of observations and iterations
    num_iters = simulation_conf["num_iters"]
    num_observations = simulation_conf["num_observations"]

    # rate of the poisson process
    poisson_rate = simulation_conf["poisson_rate"]

    # getting gaussian mixture model
    gmm_components = simulation_conf["gmm_components"]

    # getting rotation matrix
    R_theta = simulation_conf["R_theta"]

    # getting number of rotations
    K = simulation_conf["K"]

    # getting state dimension
    dim = simulation_conf["dim"]

    # retrieving the device
    device = simulation_conf["device"]

    # defining the store for the states and observations
    state_store = torch.zeros((num_iters, dim))
    observation_store = torch.zeros((num_observations, dim))

    # sampling the first state and storing it
    X = gmm_components[0].sample()
    state_store[0, :] = X
    observation_store[0, :] = X

    # counting number of rotations
    num_rotations = 0

    # progress bar
    time = tqdm(range(num_iters))


    # running the simulation
    for t in range(num_iters):
        # sampling noise and scaling it
        xi = torch.randn(2, device = device)
        noise = torch.sqrt(2*delta_t)*xi

        # retrieving pdf of mixture component
        current_rotation = num_rotations%K
        pdf = gmm_components[current_rotation]

        # computing gradient of log likelihood and scaling it
        grad_p = grad_log_pdf(X, pdf, device)
        grad_p = grad_p * delta_t

        # computing the update
        X = X + grad_p + noise

        # sampling the rotation
        flag = torch.rand(1, device = device)
        if flag < poisson_rate*delta_t:
            num_rotations += 1
            X = torch.matmul(R_theta(1),X)

        # saving the states to the store
        state_store[t, :] = X
        if t%observation_interval == 0:
            index = t // observation_interval
            observation_store[index, :] = X
        time.update()
    return state_store, observation_store

# function for getting the angular representation of a 2D vector
def vec2angle(x, y):
    # normalize vectors
    norm = torch.sqrt(x**2 + y**2)
    x_, y_ = x/norm, y/norm
    # computing angle
    theta = torch.arctan2(y_, x_)
    return theta

# function for safely concatenating two image tensor of either 3 or 4 (batched) dimensions along the 
# channel dimension to create the conditioning context described in the paper
def concat_along_channel_dim(tensor_0, tensor_1):
    ##################################################################
    ################## INPUT VERIFICATION ############################
    ##################################################################
    state_ndims = tensor_0.dim()
    context_ndims = tensor_1.dim()
    assert state_ndims == context_ndims, f"{state_ndims=} and {context_ndims=} differ ({tensor_0.shape=}, {tensor_1.shape})"
    assert state_ndims in [3, 4], f"{state_ndims=} is expected to either be 3 (C, H, W) or 4 (B, C, H, W)"
    ##################################################################
    ################## INPUT CONCATENATION ###########################
    ##################################################################
    # if we only have 3 dimensions, they 
    # are supposed to be (C, H, W) so the 
    # channel index will be 0
    if state_ndims == 3:
        channel_dim = 0
    # this will only be in case we have 4 dims
    # because we passed the assertion above
    else:  
        channel_dim = 1
    tensor_cat = torch.cat([tensor_0, tensor_1], dim = channel_dim)
    return tensor_cat

# function for adding a batch dimension 
# to an input batch dictionary in case
# it is not already present, as inferred 
# by the num_input_dims argument
def add_batch_dimension(batch, num_input_dims):
    batch_copy = dict()
    for key, value in batch.items():
        # checking if the batch does not 
        # already have the batch dimension
        # in case is already batched, do nothing
        if value.dim() == num_input_dims:
            # adding the batch dimension
            value = torch.unsqueeze(value, dim = 0)
        # if this is not the case perform sanity check to
        # verify that the number of dimension is has already the 
        # needed value of num_inpu_dims + 1 (batch dimension)
        assert(value.dim() == num_input_dims + 1), f"the tensor has an unexpected number of dimensions ({value.shape=})"
        # storing it back to the batch dictionary
        batch_copy[key] = value
    return batch_copy