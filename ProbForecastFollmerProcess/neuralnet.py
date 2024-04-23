import torch
import torch.nn.functional as F
from torch import nn

class MLP(torch.nn.Module):
    def __init__(self, input_dim, layer_widths, activate_final = False, activation_fn = torch.nn.ReLU()):
        """
        Parameters
        ----------    
        input_dim : int specifying input_dim of input 

        layer_widths : list specifying width of each layer 
            (len is the number of layers, and last element is the output input_dim)

        activate_final : bool specifying if activation function is applied in the final layer

        activation_fn : activation function for each layer        
        """
        super(MLP, self).__init__()
        layers = []
        prev_width = input_dim
        for layer_width in layer_widths:
            layers.append(torch.nn.Linear(prev_width, layer_width))
            prev_width = layer_width
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.layers = torch.nn.ModuleList(layers)
        self.activate_final = activate_final
        self.activation_fn = activation_fn
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x

class B_Network(torch.nn.Module):

    def __init__(self, dimension_state, config):
        """
        Parameters
        ----------
        Network taking as input the state and the interpolant at a given time
        dimension_state : int specifying state dimension
        config : dict containing      
            layers : list specifying width of each layer 
        """
        super().__init__()
        input_dimension = dimension_state*2 + 1# interpolant + state + time     
        layers = config['layers']
        self.standardization = config['standardization']
        self.net = MLP(
            input_dimension, 
            layer_widths = layers + [dimension_state], # output should be the dimension of the state
        )

    def forward(self, interpolant, t, state):
        """
        Parameters
        ----------
        state : (N, d)
        
        interpolant : (N, d)

        time: (1) or (N, 1)
                        
        Returns
        -------    
        out :  output (N, d)
        """
        
        N = state.shape[0]
        if len(t.shape) == 0:
            t_ = t.repeat((N, 1))
        else:
            t_ = t         
        state_c = (state - self.standardization['state_mean']) / self.standardization['state_std']
        interpolant_c = (interpolant - self.standardization['state_mean']) / self.standardization['state_std']            
        h = torch.cat([t_, state_c, interpolant_c], -1) # size (N, 1+d+p)            
        out = torch.squeeze(self.net(h)) # size (N)
        return out