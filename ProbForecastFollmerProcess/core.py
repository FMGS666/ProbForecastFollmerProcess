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
        self.train_data = data["train"] # dataset holding the train observations
        self.test_data = data["test"] # dataset holding the test observations

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
        batch_size = optim_config['batch_size']
        num_epochs = optim_config['num_epochs']
        learning_rate = optim_config['learning_rate']
        num_mc_samples = optim_config['num_mc_samples']
        # constructing data loader 
        train_data_loader = torch.utils.data.DataLoader(self.train_data, batch_size = batch_size, shuffle = True)
        # getting the number of batches
        num_batches = len(train_data_loader)
        # initializing optimizer and scheduler
        optimizer = torch.optim.AdamW(self.training_parameters, lr = learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs*num_batches)
        # optimization
        loss_values = torch.zeros(num_epochs*num_batches)
        lr_values = torch.zeros(num_epochs*num_batches)
        # iterating over the number of epochs
        for current_epoch in range(num_epochs):
            # iterating over the train data loader
            for batch_idx, (current_state, next_state) in enumerate(train_data_loader):
                # sampling the noise
                z = torch.randn(current_state.shape, device = self.device)

                # defining the time samples for monte carlo integrations
                mc_samples = torch.rand(num_mc_samples, device = self.device)

                # defining store for mc estimation of loss
                loss_store = torch.zeros(num_mc_samples, device = self.device)

                # iterating over each sample for MC integration
                for i, s in enumerate(mc_samples):

                    # computing the interpolant and velocity
                    interpolant = self.interpolant(current_state, next_state, z, s)
                    velocity = self.velocity(current_state, next_state, z, s)

                    # forward pass on the model
                    drift = self.B_net.forward(interpolant, current_state, s)
            
                    # computing loss and storing it
                    loss = F.mse_loss(drift, velocity)
                    loss_store[i] = loss

                # averaging the loss over the mc samples
                loss = torch.mean(loss_store)
                
                # backpropagation
                loss.backward()

                # optimization step and zero gradient
                optimizer.step()
                optimizer.zero_grad()

                # getting the iteration index
                index = current_epoch*num_batches + batch_idx
                # store loss 
                current_loss = loss.item()
                loss_values[index] = current_loss
        
                # scheduler step
                scheduler.step()

                # store learning rate
                current_lr = optimizer.param_groups[0]["lr"]
                lr_values[index] = current_lr
            
            # show progress
            print('Optimization epoch:', current_epoch+1, 'Learning Rate:', current_lr, 'Loss:', current_loss) 

        # output loss values and lrs
        self.loss = loss_values
        self.lrs = lr_values
    
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

    # method for performing a single sampling step on a batch of data
    def sampling_step(self, X0):
        # getting number of observations
        num_obs = X0.shape[0]
        # computing drift
        drift = self.stepsizes[0]*self.B_net(X0, X0, self.time[0])
        # sampling noise 
        eta = torch.randn((num_obs, self.dim), device = self.device)
        # computing diffusion
        diffusion = self.sigma(self.time[0])*torch.sqrt(self.stepsizes[0])*eta
        # updating state
        X = X0 + drift + diffusion

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
        return X

    def sample(self, sample_config, train = False):
        # getting number of samples
        num_samples = sample_config["num_samples"]
        # setting the target sampler function (either train or test)
        data_fun = self.train_data if train else self.test_data

        # getting the data to sample from 
        X0, X1 = data_fun[:]

        # getting number of observations
        num_obs = len(data_fun)

        # defining the store for the estimated next states
        samples_store = torch.zeros((num_samples, num_obs, self.dim))

        # iterating over the number of the samples generated for each obs
        for sample_id in range(num_samples):
            # performing sampling step
            X = self.sampling_step(X0)

            # storing observation
            samples_store[sample_id, :, :] = X
            if (sample_id + 1)%10 == 0:
                print(f"{sample_id + 1} samples generated")
        return (X0, X1), samples_store

    def sample_autoregressive(self, sample_config, train = False):
        # getting the number of autoregressive steps
        num_ar_steps = sample_config["num_ar_steps"]
        
        # setting the target sampler function (either train or test)
        data_fun = self.train_data if train else self.test_data

        # defining the store for the estimated next states
        ar_samples_store = torch.zeros((num_ar_steps, self.dim))

        # getting the first point of the dynamics to simulate autoregressively 
        current_obs, _ = data_fun[0] # shape (self.dim)
        X0 = torch.unsqueeze(current_obs, dim = 0) # shape (1, self.dim)
        
        # retrieving the ground truth path over the number of autoregressive steps
        _, gt_path = data_fun[:num_ar_steps]

        # iterating over each auto regressive step
        for ar_step in range(num_ar_steps):
            # performing sampling step
            X = self.sampling_step(X0)
            # storing sampled observation 
            ar_samples_store[ar_step, :] = X
            # setting sampled observation as starting condition
            X0 = X
            if (ar_step + 1)%50 == 0:
                print(f"{ar_step + 1} auto regressive steps taken")
        return (current_obs, gt_path), ar_samples_store