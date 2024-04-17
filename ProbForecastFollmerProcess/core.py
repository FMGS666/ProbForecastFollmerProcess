import torch    
import torch.nn.functional as F
from ProbForecastFollmerProcess.neuralnet import B_Network

def construct_time_discretization(N, device):    
    time = torch.linspace(0.0, 1.0, N + 1, device = device)
    stepsizes = (1.0 / N) * torch.ones(N, device = device)
    return (time, stepsizes)

class model(torch.nn.Module):
    def __init__(self, data, sample, state, interpolant, velocity, net_config, device = 'cpu'):
        super(model, self).__init__()
        # device
        self.device = device

        # data (callable)
        self.train_data = data["train"] # (callable) the function for retrieving the train data
        self.test_data = data["test"] # (callable) the function for retrieving the test data

        # sample
        self.N = sample["N"] # (int) the number of points for the euler discretization
        self.g = sample["g"] # (callable) the function for computing the tunable diffusion term

        # constructing discretization
        (self.time, self.stepsizes) = construct_time_discretization(self.N, device = self.device)

        # state
        self.dim = state["dim"]

        # interpolant (callables)
        self.alpha = interpolant["alpha"]
        self.beta = interpolant["beta"]
        self.sigma = interpolant["sigma"]

        # velocity (callables)
        self.alpha_dot = velocity["alpha_dot"]
        self.beta_dot = velocity["beta_dot"]
        self.sigma_dot = velocity["sigma_dot"]

        # initialize approximator network 
        self.B_net = B_Network(self.dim, net_config)
        self.B_net.to(device)
        self.training_parameters = [{"params": self.B_net.parameters()}]

    def interpolant(self, current_state, next_state, z, s):
        # getting interpolant coefficients
        alpha = self.alpha(s)
        beta = self.beta(s)
        sigma = self.sigma(s)
        # computing interpolant
        interpolant = alpha*current_state + beta*next_state + torch.sqrt(s)*sigma*z
        return interpolant

    def velocity(self, current_state, next_state, z, s):
        # getting velocity coefficients
        alpha_dot = self.alpha_dot(s)
        beta_dot = self.beta_dot(s)
        sigma_dot = self.sigma_dot(s)
        # computing velocity
        velocity = alpha_dot*current_state + beta_dot*next_state + torch.sqrt(s)*sigma_dot*z
        return velocity
    
    def train(self, optim_config):
        # optimization configuration
        minibatch = optim_config['minibatch']
        num_obs_per_batch = optim_config['num_obs_per_batch']
        N = minibatch * num_obs_per_batch
        num_iterations = optim_config['num_iterations']
        learning_rate = optim_config['learning_rate']
        num_mc_samples = optim_config['num_mc_samples']
        # initializing optimizer and scheduler
        optimizer = torch.optim.AdamW(self.training_parameters, lr = learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N*num_iterations)    

        # optimization
        loss_values = torch.zeros(num_iterations, device = self.device)
        for current_iter in range(num_iterations):
            # sampling current states and next states
            current_state, next_state = self.train_data(N)

            # defining the time samples for monte carlo integrations
            mc_samples = torch.rand(num_mc_samples, device = self.device)

            # defining store for output and velocities for current iteration
            drift_store = torch.zeros(num_mc_samples, N, self.dim, device = self.device)
            velocity_store = torch.zeros(num_mc_samples, N, self.dim, device = self.device)

            # iterating over each sample for MC integration
            for i, s in enumerate(mc_samples):
                
                # sampling the noise
                z = torch.randn(current_state.shape, device = self.device)

                # computing the interpolant and velocity
                interpolant = self.interpolant(current_state, next_state, z, s)
                velocity = self.velocity(current_state, next_state, z, s)

                # forward pass on the model
                drift = self.B_net.forward(interpolant, current_state, s)
        
                # storing drift and velocities
                drift_store[i, :, :] = drift
                velocity_store[i, :, :] = velocity
            
            # reshaping the store to (num_mc_sample*N, self.dim)
            drift_store = torch.flatten(drift_store, start_dim = 0, end_dim = 1)
            velocity_store = torch.flatten(velocity_store, start_dim = 0, end_dim = 1)

            # computing loss
            loss = F.mse_loss(drift_store, velocity_store)
            
            # backpropagation
            loss.backward()
    
            # optimization step and zero gradient
            optimizer.step()
            optimizer.zero_grad()

            # scheduler step
            scheduler.step()

            # store loss 
            current_loss = loss.item()
            loss_values[current_iter] = current_loss
            if (current_iter == 0) or ((current_iter+1) % 50 == 0):
                print('Optimization iteration:', current_iter+1, 'Loss:', current_loss)

        # output loss values
        self.loss = loss_values
    
    def score(self, interpolant, current_state, s):
        # computing learned drift
        with torch.no_grad():
            drift = self.B_net(interpolant, current_state, s)
        # computing score factors
        c = self.beta_dot(s)*interpolant + (self.beta(s)*self.alpha_dot(s) - self.beta_dot(s)*self.alpha(s))*current_state
        A = 1/(s*self.sigma(s)*(self.beta_dot(s)*self.sigma(s) - self.beta(s)*self.sigma_dot(s)))
        score = A*(self.beta(s)*drift - c)
        return score
    
    def adjusted_drift(self, interpolant, current_state, s):
        # computing learned drift
        with torch.no_grad():
            drift = self.B_net(interpolant, current_state, s)
        # computing score
        score = self.score(interpolant, current_state, s)
        # computing control drift term
        control = 0.5*(self.g(s)**2 - self.sigma(s)**2)*score
        controlled_drift = drift + control
        return controlled_drift

    def sample(self, sample_config, train = False):
        # getting the number of samples onn the 0, 1 interval
        minibatch = sample_config['minibatch']
        num_obs_per_batch = sample_config['num_obs_per_batch']
        num_samples_per_obs = sample_config["num_samples_per_obs"]
        num_obs = minibatch * num_obs_per_batch
        
        # setting the target sampler function (either train or test)
        data_fun = self.train_data if train else self.test_data

        # getting the data to sample from 
        current_states, next_states = data_fun(num_obs)

        # defining the store for the estimated next states
        samples_store = torch.zeros((num_samples_per_obs, num_obs, self.dim))

        # iterating over the number of the samples generated for each obs
        for sample_id in range(num_samples_per_obs):
            # computing first observation and storing it
            X0 = current_states
            X1 = next_states
            # computing drift
            drift = self.stepsizes[0]*self.B_net(X0, X0, self.time[0])
            # sampling noise 
            eta = torch.randn((num_obs, self.dim), device = self.device)
            # computing diffusion
            diffusion = self.sigma(self.time[0])*torch.sqrt(self.stepsizes[0])*eta
            # updating state
            X = X0 + drift + diffusion
            # storing observations
            samples_store[0, :] = X

            # iterating over each step of the euler discretization
            for n in range(1, self.N):
                # gettng the stepsize
                delta_s = self.stepsizes[n]
                s = self.time[n]

                # computing adjusted drift
                drift = delta_s*self.adjusted_drift(X, X0, s)

                # sampling noise
                eta = torch.randn((num_obs, self.dim), device = self.device)

                # computing diffusion term
                diffusion = self.g(s)*torch.sqrt(delta_s)*eta

                # euler step
                X = X + drift + diffusion

            # storing observation
            samples_store[sample_id, :, :] = X
            if (sample_id + 1)%10 == 0:
                print(f"{sample_id + 1} samples generated")
        return (X0, X1), samples_store