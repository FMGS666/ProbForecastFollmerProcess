import torch    
import torch.nn.functional as F
from ProbForecastFollmerProcess.neuralnet import B_Network

def construct_time_discretization(N, device):    
    time = torch.linspace(0.0, 1.0, N + 1, device = device)
    stepsizes = (1.0 / N) * torch.ones(N, device = device)
    return (time, stepsizes)

class model(torch.nn.Module):
    def __init__(self, backbone, data, sample, state, interpolant, velocity, optim, device = 'cpu', debug = False, verbose = 1, random_ar_context = False):
        super(model, self).__init__()
        # device
        self.device = device

        # data (callable)
        self.train_data = data["train"] # dataset holding the train observations
        self.test_data = data["test"] # dataset holding the test observations

        # sample
        self.num_euler_steps = sample["num_euler_steps"] # (int) the number of points for the euler discretization
        self.g = sample["g"] # (callable) the function for computing the tunable diffusion term
        # setting the optional random autoregressive context
        # that is needed to correctly update the batch in case of random
        # observation picked from the past when running autoregressive sampling
        # This should be set to True when running the example on video generation, while to False (default) for jump diffusion
        # If false, we assume current_state == conditioning_state
        self.random_ar_context = random_ar_context

        # constructing discretization
        (self.time, self.stepsizes) = construct_time_discretization(self.num_euler_steps, device = self.device)

        # state spatial dimensions
        self.spatial_dims = state["spatial_dims"]

        # interpolant (callables)
        self.alpha = interpolant["alpha"]
        self.beta = interpolant["beta"]
        self.sigma = interpolant["sigma"]

        # velocity (callables)
        self.alpha_dot = velocity["alpha_dot"]
        self.beta_dot = velocity["beta_dot"]
        self.sigma_dot = velocity["sigma_dot"]

        # initialize backbone approximator network
        # attribute (an instance of torch.nn.Module)
        self.backbone = backbone
        self.backbone.to(self.device)
        self.training_parameters = [{"params": self.backbone.parameters()}]


        # optimization configuration
        self.batch_size = optim['batch_size']
        self.num_epochs = optim['num_epochs']
        self.learning_rate = optim['learning_rate']
        self.num_mc_samples = optim['num_mc_samples']
        self.max_num_grad_steps = optim['max_num_grad_steps']


        # constructing data loader 
        self.train_data_loader = torch.utils.data.DataLoader(self.train_data, batch_size = self.batch_size, shuffle = True)
        # getting the number of batches
        self.num_batches = len(self.train_data_loader)
        # getting the number of optimization iterations
        self.num_optim_iters = self.num_epochs*self.num_batches

        # initializing optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.training_parameters, lr = self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.num_optim_iters)

        # store for losses and learning rates
        self.loss_values = torch.zeros(self.num_optim_iters)
        self.lrs_values = torch.zeros(self.num_optim_iters)

        # debugging
        self.debug = debug

        # verbose 
        self.verbose = verbose

    # method for computing the stochastic interpolant of (15)
    # input: (X0) current state, (X1) next state, (z) random noise, (s) time
    # returns: the empirical interpolant evaluated on the current input
    def interpolant(self, X0, X1, z, s):
        # getting interpolant coefficients
        alpha = self.alpha(s)
        beta = self.beta(s)
        sigma = self.sigma(s)
        # computing interpolant
        interpolant = alpha*X0 + beta*X1 + torch.sqrt(s)*sigma*z
        return interpolant

    # method for computing the stochastic velocity of (15)
    # input: (X0) current state, (X1) next state, (z) random noise, (s) time
    # returns: the empirical velocity evaluated on the current input
    def velocity(self, X0, X1, z, s):
        # getting velocity coefficients
        alpha_dot = self.alpha_dot(s)
        beta_dot = self.beta_dot(s)
        sigma_dot = self.sigma_dot(s)
        # computing velocity
        velocity = alpha_dot*X0 + beta_dot*X1 + torch.sqrt(s)*sigma_dot*z
        return velocity

    # method for computing the score function of (10)
    # input: (drift) the estimated drift, (X) interpolant, (X0) current state, (s) time
    # returns: the score functon evaluated on the initial solution and the current interpolant
    def score(self, drift, X, X0, s):
        # computing score factors
        c = self.beta_dot(s)*X + (self.beta(s)*self.alpha_dot(s) - self.beta_dot(s)*self.alpha(s))*X0
        A = 1/(s*self.sigma(s)*(self.beta_dot(s)*self.sigma(s) - self.beta(s)*self.sigma_dot(s)))
        score = A*(self.beta(s)*drift - c)
        return score
    
    # method for computing the adjusted drift of (8)
    # input: (X) interpolant, (s) time, (X0) current state, (Xc) conditioning state
    # returns: the estimated drift adjusted by the custom diffusion process
    def adjusted_drift(self, X, s, X0, Xc):
        # computing learned drift
        with torch.no_grad():
            drift = self.backbone(X, s, Xc)
        # computing score
        score = self.score(drift, X, X0, s)
        # computing control drift term
        control = 0.5*(self.g(s)**2 - self.sigma(s)**2)*score
        adjusted_drift = drift + control
        return adjusted_drift

    # method for handling the batch update 
    # in the autoregressive sampling scheme
    def batch_ar_update(self, ar_step, ar_samples_store):
        # getting the last sampled state
        current_state = ar_samples_store[ar_step, :]
        conditioning_state = current_state
        if self.random_ar_context:
            # sampling random time index in range {0, ..., ar_step - 1}
            random_context_index = torch.randint(high = ar_step, size = (1, ))
            # retrieving random context state 
            random_context_state = ar_samples_store[random_context_index, :]
            # concatenating the current state and the random context state
            # along channel dimension to construct the conditioning state
            conditioning_state = torch.cat([conditioning_state, random_context_state], dim = 1) # (b, c*2, ...)
        # constructing the batch dictionary 
        # again we need to add the batch dimension as we assume for now the 
        # autoregressive sampling is done each time from a single observation
        current_state = torch.unsqueeze(current_state, dim = 0)
        conditioning_state = torch.unsqueeze(conditioning_state, dim = 0)
        # constructing the batch dictionary while moving tensors to device
        batch = {"current_state": current_state.to(self.device), "conditioning_state": conditioning_state.to(self.device)}
        return batch

    # method for performing a forward and backward pass on a training batch
    # input: the batch on which to perform the training step
    # returns: the monte carlo estimation of the loss of equation (14)
    def train_step(self, batch):
        ################### READING BATCH DICTIONARY ######################
        # retrieving current state, next state and conditioning state
        X0 = batch["current_state"]
        X1 = batch["next_state"]
        Xc = batch["conditioning_state"]

        ###################################################################
        ########### FORWARD PASS AND MC ESTIMATION OF LOSS ################
        ###################################################################

        # sampling the noise
        z = torch.randn_like(X0, device = self.device)
        # defining the time samples for monte carlo integrations
        mc_samples = torch.rand(self.num_mc_samples, device = self.device)
        # defining store for mc estimation of loss
        loss_store = torch.zeros(self.num_mc_samples, device = self.device)

        # iterating over each sample for MC integration
        for i, s in enumerate(mc_samples):
            # computing the interpolant and velocity
            X = self.interpolant(X0, X1, z, s)
            R = self.velocity(X0, X1, z, s)

            # forward pass on the model
            drift = self.backbone.forward(X, s, Xc)
    
            # computing loss and storing it
            loss = F.mse_loss(drift, R)
            loss_store[i] = loss

        ###################################################################
        ######################## BACKWARD PASS ############################
        ###################################################################

        # averaging the loss over the mc samples
        loss = torch.mean(loss_store)

        # backpropagation
        loss.backward()
        # optimization step and zero gradient
        self.optimizer.step()
        self.optimizer.zero_grad()
        # scheduler step
        self.scheduler.step()
        return loss.item()

    # method for performing a single sampling step on a batch of data
    # input: the batch on which to perform the sampling step
    # returns: single realization of a sample obtained with algorithm (2)
    def sampling_step(self, batch):
        ################### READING BATCH DICTIONARY ######################
        # retrieving current state, next state and conditioning state
        X0 = batch["current_state"]
        Xc = batch["conditioning_state"]
        # getting number of observations
        num_obs = X0.shape[0]
        ###################################################################
        ########### INTEGRATING USING EULER DISCRETIZATION ################
        ###################################################################
        # first euler step is different as using the adjusted drift
        # with s = 0 leads to numerical singularities

        # computing drift
        drift = self.stepsizes[0]*self.backbone(X0, self.time[0], Xc)
        # sampling noise 
        eta = torch.randn(X0.shape, device = self.device)
        
        # computing diffusion
        diffusion = self.sigma(self.time[0])*torch.sqrt(self.stepsizes[0])*eta
        # updating state
        X = X0 + drift + diffusion
        ###################################################################
        ###################################################################

        # iterating over each step of the euler discretization
        for n in range(1, self.num_euler_steps):
            # gettng time and stepsize
            delta_s = self.stepsizes[n]
            s = self.time[n]

            # computing adjusted drift
            drift = delta_s*self.adjusted_drift(X, s, X0, Xc)

            # sampling noise
            eta = torch.randn(X0.shape, device = self.device)

            # computing diffusion term
            diffusion = self.g(s)*torch.sqrt(delta_s)*eta

            # euler step
            X = X + drift + diffusion
        return X

    def train(self):
        # initializing counter of gradient steps
        num_grad_steps = 0
        continue_training = True
        # one liner function for checking the condition of stop train
        stop_train = lambda: self.max_num_grad_steps and num_grad_steps > self.max_num_grad_steps
        # iterating over the number of epochs
        for current_epoch in range(self.num_epochs):
            # iterating over the train data loader
            for batch_idx, batch in enumerate(self.train_data_loader):
                # showing optional debugging message        
                if self.debug:
                    for k, v in batch.items():
                        print(k, v.shape)        

                # performing full training step
                current_loss = self.train_step(batch)
            
                # retrive learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]

                # getting the iteration index
                index = current_epoch*self.num_batches + batch_idx

                # store learning rate and loss 
                self.loss_values[index] = current_loss
                self.lrs_values[index] = current_lr

                # updating counter of gradient steps
                num_grad_steps += 1

                # displaying progress
                if self.verbose > 1:
                    print(f"epoch: {current_epoch+1}/{self.num_epochs}, batch: {batch_idx}/{len(self.train_data_loader)}, loss: {current_loss}, learning_rate: {current_lr}")
                # checking if the maximum number of gradient steps has been reached
                # and breaking batching loop in case
                if stop_train():
                    break
            # displaying progress
            if self.verbose > 0:
                print(f"epoch: {current_epoch+1}/{self.num_epochs}, loss: {current_loss}, learning_rate: {current_lr}")
            # checking if the number of gradient steps has been reached
            # and breaking epoch loop in case
            if stop_train():
                break

    def sample(self, sample_config, train = False):
        # getting number of samples and observations
        num_samples = sample_config["num_samples"]
        num_obs = sample_config["num_obs"]
        # setting the target sampler function (either train or test)
        data_fun = self.train_data if train else self.test_data
        # retrieving batch dictionary
        batch = data_fun[:num_obs]
        # showing optional debugging message
        if self.debug:
            for k, v in batch.items():
                print(k, v.shape)
        # defining the store for the estimated next states
        samples_store = torch.zeros((num_samples, *batch["current_state"].shape))
        # iterating over the number of the samples generated for each obs
        for sample_id in range(num_samples):
            # performing sampling step
            X = self.sampling_step(batch)
            # storing observation
            samples_store[sample_id, :, :] = X
            # displaying progress
            if self.verbose > 0 and (sample_id + 1)%50 == 0:
                print(f"{sample_id + 1} samples generated")
        output = {"current_states": batch["current_state"], "next_state": batch["next_state"], "sampled_states": samples_store}
        return output

    def sample_autoregressive(self, sample_config, train = False):
        # getting the number of autoregressive steps
        num_ar_steps = sample_config["num_ar_steps"]
        
        # setting the target sampler function (either train or test)
        data_fun = self.train_data if train else self.test_data

        # defining the store for the estimated next states
        ar_samples_store = torch.zeros((num_ar_steps, *self.spatial_dims))

        # retrieving batch dictionary
        batch = data_fun[0] 
        # since we are only getting the first element we 
        # are losing the batch dimension so we need to
        # add it for each value in the batch dictionary
        for key, value in batch.items():
            # adding the batch dimension
            value = torch.unsqueeze(value, dim = 0)
            # storing it back to the batch dictionary
            batch[key] = value
        # showing optional debugging message
        if self.debug:
            for k, v in batch.items():
                print(k, v.shape)
        # retrieving the ground truth path over the number of autoregressive steps
        gt_path = data_fun[:num_ar_steps]["next_state"]

        # iterating over each auto regressive step
        for ar_step in range(num_ar_steps):
            # performing sampling step
            X = self.sampling_step(batch)
            # storing sampled observation 
            ar_samples_store[ar_step, :] = X
            # updating batch
            batch = self.batch_ar_update(ar_step, ar_samples_store)
            # displaying progress
            if self.verbose > 0 and (ar_step + 1)%50 == 0:
                print(f"{ar_step + 1} auto regressive steps taken")
        output = {"initial_condition": data_fun[0]["current_state"], "gt_path": gt_path, "ar_path": ar_samples_store}
        return output