import math
import torch
import sys
sys.path.append('/home/tachennf/Documents/MRI-pH-Hypoxia/torchcde')
import torchcde
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


######################
# A CDE model looks like
#
# z_t = z_0 + \int_0^t f_\theta(z_s) dX_s
#
# Where X is your data and f_\theta is a neural network. So the first thing we need to do is define such an f_\theta.
# That's what this CDEFunc class does.
# Here we've built a small single-hidden-layer neural network, whose hidden layer is of width 128.
######################
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. It's the number of offsets in the data.
        # hidden_channels is the number of channels for z_t.
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


class FTheta(torch.nn.Module):
    # Based on the table from Kuppens et al. 
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. It's the batch size.
        # hidden_channels is the number of channels for z_t. It's the number of parameters to estimate.
        ######################
        super(FTheta, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, 256)
        self.linear3 = torch.nn.Linear(256, 512)
        self.linear4 = torch.nn.Linear(512, 256)
        self.linear5 = torch.nn.Linear(256, 128)
        self.linear6 = torch.nn.Linear(128, input_channels * hidden_channels)

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        z = z.relu()
        z = self.linear3(z)
        z = z.relu()
        z = self.linear4(z)
        z = z.relu()
        z = self.linear5(z)
        z = z.relu()
        z = self.linear6(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z
    
class LTheta2(torch.nn.Module):
    # Based on the table from Kuppens et al. 
    def __init__(self, input_channels, hidden_channels, output_channels):
        ######################
        # input_channels is the number of input channels in the data X. It's the batch size.
        # hidden_channels is the number of channels for z_t. It's the number of parameters to estimate.
        # output_channels is the number of output channels in the prediction. 2 for binary classification, m for the number of parameters to estimate in regression.
        ######################
        super(LTheta2, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, hidden_channels) # input_channels *hidden_channels
        self.linear2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.linear3 = torch.nn.Linear(hidden_channels, output_channels)
        self.batchnorm = torch.nn.BatchNorm1d(hidden_channels)

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, input_channels*hidden_channels)
        z = self.linear1(z)
        z = self.batchnorm(z)
        z = z.relu()

        z = self.linear2(z)
        z = self.batchnorm(z)
        z = z.relu()
        z = self.linear2(z)
        z = self.batchnorm(z)
        z = z.relu()
        z = self.linear2(z)
        z = self.batchnorm(z)
        z = z.relu()
        z = self.linear2(z)
        z = self.batchnorm(z)
        z = z.relu()        

        z = self.linear3(z)

        return z
    

######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, interpolation="cubic"):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        # output_channels is the number of output channels in the prediction. 2 for binary classification, m for the number of parameters to estimate in regression.
        super(NeuralCDE, self).__init__()
        self.initial = torch.nn.Linear(input_channels, hidden_channels) # ltheta1, the initial value of z_t
        self.func = FTheta(input_channels, hidden_channels) # func is ftheta, the function that defines the CDE
        self.readout = LTheta2(input_channels, hidden_channels, output_channels) # ltheta2, the readout function that maps the final value of z_t to the output
        self.interpolation = interpolation

    def forward(self, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0) # ltheta1 is initial 

        ######################
        # Actually solve the CDE.
        ######################
        z_T = torchcde.cdeint(X=X, # This will be a tensor of shape (..., len(t)-1, hidden_channels).
                              z0=z0,
                              func=self.func, # func is ftheta 
                              t=X.interval) 

        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value (not z0),
        # and then apply a linear map.
        ######################
        z_T = z_T[:, 1]
        pred_y = self.readout(X.interval, z_T) # readout is ltheta2
        return pred_y
    
def ivim_model(b, S0, f, d_slow, d_fast, torch_based: bool = False):
    '''Standard IVIM model with both fast and slow diffusion coefficients as fitting parameters.

    Parameters:
    b (torch.Tensor): b-value tensor of shape [batch_size, num_b_values]
    S0 (torch.Tensor): Signal intensity at b=0, tensor of shape [batch_size]
    f (torch.Tensor): Perfusion fraction, tensor of shape [batch_size]
    d_slow (torch.Tensor): Slow diffusion coefficient, tensor of shape [batch_size]
    d_fast (torch.Tensor): Fast diffusion coefficient, tensor of shape [batch_size]

    Returns:
    torch.Tensor: Signal intensity at b, tensor of shape [batch_size, num_b_values]
    '''
    if torch_based:
        # Reshape S0, f, d_slow, and d_fast to [batch_size, 1, 1] for broadcasting
        S0 = S0.view(-1, 1, 1)
        f = f.view(-1, 1, 1)
        d_slow = d_slow.view(-1, 1, 1)
        d_fast = d_fast.view(-1, 1, 1)

        # Perform element-wise operations
        return S0 * (f * torch.exp(-b * d_fast) + (1 - f) * torch.exp(-b * d_slow))
    else:
        return S0 * (f * np.exp(-b * d_fast) + (1 - f) * np.exp(-b * d_slow))

def get_random_b_values(n_b_values: int = 7):
    '''Generate random b-values for IVIM fitting.

    Parameters:
    n_b_values (int): Number of b-values to generate

    Returns:
    np.ndarray: Array of random b-values
    '''
    b_values = [np.array([0])]    
    b_values.append(np.random.uniform(10, 50, size=int((n_b_values - 1)/2)))
    b_values.append(np.random.uniform(50, 100, size=int((n_b_values - 1)/4)))
    b_values.append(np.random.uniform(100, 800, size=int((n_b_values - 1)/3)))
    b_values = np.sort(np.concatenate(b_values))
    return b_values

def get_ivim_data(param_ranges: list, n_b_values: int = 7, batch_size: int = 100):
    '''Generate synthetic IVIM data for training and testing.

    Parameters:
    param_ranges (list): range of the parameters to simulate
    n_b_values (int): Number of b-values to generate
    sampling_size (int): Number of samples to generate for each combination of parameters

    Returns:
    X (np.ndarray): Generated IVIM data with shape (num_samples, n_b_values)
    y (np.ndarray): Corresponding parameters for each sample with shape (num_samples, 4)
    b_values (np.ndarray): Array of b-values used in the IVIM model
    '''

    S0_values = np.random.uniform(param_ranges[0][0], param_ranges[0][1], size=batch_size)
    print(S0_values.shape)
    f_values = np.random.uniform(param_ranges[1][0], param_ranges[1][1], size=batch_size)
    D_slow_values = np.random.uniform(param_ranges[2][0], param_ranges[2][1], size=batch_size)
    D_fast_values = np.random.uniform(param_ranges[3][0], param_ranges[3][1], size=batch_size)
    noise_levels = [5, 10, 20, 30, 40]

    # X is a tensor of observations, of shape (batch=128, sequence=100, channels=3)

    X = np.zeros((batch_size, S0_values.size*D_slow_values.size*D_fast_values.size*f_values.size*len(noise_levels), n_b_values))
    y = np.zeros((batch_size, 4)) # 4 because it's ivim data 
    b_values = np.zeros((batch_size, S0_values.size*D_slow_values.size*D_fast_values.size*f_values.size*len(noise_levels), n_b_values))
    print("Shape of X:", X.shape, "Shape of y:", y.shape, "Shape of b_values:", b_values.shape)
    
    i = 0 
    while i < batch_size:
        b_values_random = get_random_b_values(n_b_values)
        for S0 in S0_values:
            for D_slow in D_slow_values:
                for D_fast in D_fast_values:
                    for f in f_values:
                        for n in noise_levels:
                            b_values[i, :] = b_values_random
                            signal = ivim_model(b_values[i, :], S0, f, D_slow, D_fast, torch_based=False)
                            noisy_signal = np.sqrt((signal + np.random.normal(0, signal[0]/n))**2 + np.random.normal(0, signal[0]/n)**2)
                            X[i, :] = noisy_signal
                            y[i, 0] = S0
                            y[i, 1] = f
                            y[i, 2] = D_slow
                            y[i, 3] = D_fast

        i += 1
                        
    return X, y, b_values

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, coeffs, X, y, b_values):
        self.coeffs = coeffs
        self.X = X
        self.y = y
        self.b_values = b_values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.coeffs[idx], self.X[idx], self.y[idx], self.b_values[idx]
    
def rescale_out_params(param_ranges: list, y_pred: torch.Tensor) -> torch.Tensor: 
    '''Rescale output parameters of the network as y_scaled = y_min + sigmoid(y_pred)(y_max - y_min). 
    
    Parameters
    param_ranges (list): list of the min and max values used to simulate parameters
    y_pred (torch.Tensor): predictions of the neural network
    
    Returns 
    y_scaled (torch.Tensor): rescaled predictions 
    '''
    y_scaled = torch.zeros(y_pred.shape)
    for i in range(y_pred.shape[-1]): 
        y_scaled[:, i] = param_ranges[i][0] + torch.sigmoid(y_pred[:, i])(param_ranges[i][1] - param_ranges[i][0])

    return y_scaled

def main(param_ranges: list, num_epochs: int = 30, n_b_values: int = 7, batch_size: int = 50):
    device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    X, y, b_values = get_ivim_data(param_ranges, n_b_values=n_b_values, batch_size=batch_size)
    X_train, X_test, y_train, y_test, b_values_train, b_values_test = train_test_split(X, y, b_values, test_size=0.33, random_state=42)
    X_train = torch.Tensor(X_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    b_values_train = torch.Tensor(b_values_train).to(device)

    ######################
    # input_channels=3 because we have both the horizontal and vertical position of a point in the spiral, and time.
    # hidden_channels=8 is the number of hidden channels for the evolving z_t, which we get to choose.
    # output_channels=1 because we're doing binary classification.
    ######################
    # TODO: correct batchnorm in NeuralCDE, might need 2D 
    model = NeuralCDE(input_channels=n_b_values, hidden_channels=100, output_channels=4).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    ######################
    # Now we turn our dataset into a continuous path. We do this here via Hermite cubic spline interpolation.
    # The resulting `train_coeffs` is a tensor describing the path.
    # For most problems, it's probably easiest to save this tensor and treat it as the dataset.
    ######################
    ########### TRAIN ##########
    train_coeffs = torchcde.natural_cubic_spline_coeffs(X_train)
    print(train_coeffs.shape, X_train.shape, y_train.shape, b_values_train.shape)

    train_dataset = CustomDataset(train_coeffs, X_train, y_train, b_values_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch_coeffs, batch_x, batch_y, batch_b_values = batch
            y_pred = model(batch_coeffs) # .squeeze(-1)
            s_pred = ivim_model(batch_b_values, y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3], torch_based=True)
            loss = torch.nn.functional.mse_loss(y_pred, batch_y) + 1/n_b_values * torch.sum(torch.pow((batch_x-s_pred), 2)) # Add a physics informed loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))

    ########### TEST ##########
    X_test = torch.Tensor(X_test).to(device)
    y_test = torch.Tensor(y_test).to(device)
    test_coeffs = torchcde.natural_cubic_spline_coeffs(X_test) ## Apply splines
    y_pred = model(test_coeffs).squeeze(-1)
    y_scaled = rescale_out_params(param_ranges, y_pred) 
    squared_error = torch.sum(torch.pow((y_scaled - y_test), 2))
    print('Test MSE: {}'.format(squared_error))


if __name__ == '__main__':
    param_ranges = [[0.95, 1.05], [0.03, 0.25], [0.00035, 0.003], [0.05, 0.01]] # S0, f, D, D*
    main(param_ranges, batch_size=8)