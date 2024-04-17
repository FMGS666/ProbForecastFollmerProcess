import torch

from tqdm.notebook import tqdm

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

# function for getting the angular representation of a vector
def vec2angle(x, y):
    # normalize vectors
    norm = torch.sqrt(x**2 + y**2)
    x_, y_ = x/norm, y/norm
    # computing angle
    theta = torch.arctan2(y_, x_)
    return theta