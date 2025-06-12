import math
import torch
import sys
sys.path.append('/home/tachennf/Documents/MRI-pH-Hypoxia/torchcde')
import torchcde
import numpy as np

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
    


######################
# Now we need some data.
# Here we have a simple example which generates some spirals, some going clockwise, some going anticlockwise.
######################
def get_data(num_timepoints=100):
    t = torch.linspace(0., 4 * math.pi, num_timepoints)

    start = torch.rand(128) * 2 * math.pi
    x_pos = torch.cos(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos[:64] *= -1
    y_pos = torch.sin(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos += 0.01 * torch.randn_like(x_pos)
    y_pos += 0.01 * torch.randn_like(y_pos)
    ######################
    # Easy to forget gotcha: time should be included as a channel; Neural CDEs need to be explicitly told the
    # rate at which time passes. Here, we have a regularly sampled dataset, so appending time is pretty simple.
    ######################
    X = torch.stack([t.unsqueeze(0).repeat(128, 1), x_pos, y_pos], dim=2)
    y = torch.zeros(128)
    y[:64] = 1

    perm = torch.randperm(128)
    X = X[perm]
    y = y[perm]

    ######################
    # X is a tensor of observations, of shape (batch=128, sequence=100, channels=3)
    # y is a tensor of labels, of shape (batch=128,), either 0 or 1 corresponding to anticlockwise or clockwise
    # respectively.
    ######################
    return X, y

def ivim_model(b, S0, f, d_slow, d_fast):
    '''Standard IVIM model with both fast and slow diffusion coefficients as fitting parameters.

    Parameters:
    b (float): b-value
    S0 (float): Signal intensity at b=0
    f (float): perfusion fraction
    d_slow (float): slow diffusion coefficient
    d_fast (float): fast diffusion coefficient

    Returns:
    float: Signal intensity at b
    '''
    return S0 * (f * np.exp(-b*d_fast) + (1-f) * np.exp(-b*d_slow))

def get_ivim_data(n_b_values: int = 7, sampling_size: int = 100):
    D_slow_values = np.random.uniform(0.00035, 0.003, size=sampling_size)
    D_fast_values = np.random.uniform(0.05, 0.01, size=sampling_size)
    f_values = np.random.uniform(0.03, 0.25, size=sampling_size)
    noise_levels = [5, 10, 20, 30, 40]
    b_values = [np.array([0])]    
    b_values.append(np.random.uniform(10, 50, size=int((n_b_values - 1)/2)))
    b_values.append(np.random.uniform(50, 100, size=int((n_b_values - 1)/4)))
    b_values.append(np.random.uniform(100, 800, size=int((n_b_values - 1)/3)))
    b_values = np.sort(np.concatenate(b_values))
    print(b_values)
    X = np.zeros((D_slow_values.size*D_fast_values.size*f_values.size*len(noise_levels), n_b_values))

    c = 0
    for D_slow in D_slow_values:
        for D_fast in D_fast_values:
            for f in f_values:
                for n in noise_levels:
                    X[c, :] = ivim_model(b_values, 1.0, f, D_slow, D_fast) 
                    c += 1
                    for i in range(b_values.size): 
                        X[c, i] = np.sqrt((X[c, i] + np.random.normal(0, X[c, 0]/n, size=1))**2 + np.random.normal(0, X[c, 0]/n, size=1)**2)
                        
    return X, b_values

def main(num_epochs=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    train_X, train_y = get_data()
    train_X = train_X.to(device)
    train_y = train_y.to(device)
    ######################
    # input_channels=3 because we have both the horizontal and vertical position of a point in the spiral, and time.
    # hidden_channels=8 is the number of hidden channels for the evolving z_t, which we get to choose.
    # output_channels=1 because we're doing binary classification.
    ######################
    model = NeuralCDE(input_channels=3, hidden_channels=8, output_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    ######################
    # Now we turn our dataset into a continuous path. We do this here via Hermite cubic spline interpolation.
    # The resulting `train_coeffs` is a tensor describing the path.
    # For most problems, it's probably easiest to save this tensor and treat it as the dataset.
    ######################
    train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_X)

    train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch_coeffs, batch_y = batch
            pred_y = model(batch_coeffs).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))

    test_X, test_y = get_data()
    test_X = test_X.to(device)
    test_y = test_y.to(device)
    test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_X)
    pred_y = model(test_coeffs).squeeze(-1)
    binary_prediction = (torch.sigmoid(pred_y) > 0.5).to(test_y.dtype)
    prediction_matches = (binary_prediction == test_y).to(test_y.dtype)
    proportion_correct = prediction_matches.sum() / test_y.size(0)
    print('Test Accuracy: {}'.format(proportion_correct))


if __name__ == '__main__':
    get_ivim_data()
    quit()
    main()