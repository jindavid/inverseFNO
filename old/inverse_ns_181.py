import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt
# import seaborn as sns; sns.set_theme()

import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *
import random
import time

torch.manual_seed(0)
np.random.seed(0)

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

################################################################
#  Load data
################################################################

data = np.load('/home2/jindawei/fourier_neural_operator/data/KFvorticity_Re40_N200_T500.npy')
data = torch.tensor(data, dtype=torch.float)
print(data.shape)

################################################################
#  Inverse configurations
################################################################

sub = 1
s = int(data.shape[-1] / sub)

learning_rate = 0.1
scheduler_step = 100
scheduler_gamma = 0.5
epochs = 10000

################################################################
# load data
################################################################

length = data.shape[0] * data.shape[1]

dataset = data.reshape(length, 64, 64)
T = random.randint(0, length-1)

x_groundtruth = dataset[T,::sub,::sub]
y = dataset[T+1,::sub,::sub]

grids = []
grids.append(np.linspace(0, 1, s))
grids.append(np.linspace(0, 1, s))
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
grid = grid.reshape(1,s,s,2)
grid = torch.tensor(grid, dtype=torch.float)
x_groundtruth = torch.cat([x_groundtruth.reshape(1,s,s,1), grid.repeat(1,1,1,1)], dim=3)

################################################################
# load model
################################################################

PATH = 'model/KF_NS_2D_N100_ep50_m12_w32_s64'
model = torch.load(PATH)

#randomly generate parameters
x = torch.rand(x_groundtruth.shape, requires_grad=True, device="cuda")
# x = torch.norm(F.normalize(x_groundtruth, p=2, dim=1)).weight * torch.rand(x_groundtruth.shape, requires_grad=True).weight
y, x_groundtruth = y.cuda(), x_groundtruth.cuda()

optimizer = torch.optim.SGD([x], lr=learning_rate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
myloss = nn.MSELoss()

t0 = time.time()

for i in range(epochs):

    optimizer.zero_grad()
    output = model(x)
    loss = myloss(output.view(1, -1), y.view(1, -1))
    loss_groundtruth = myloss(x.view(1, -1), x_groundtruth.view(1, -1))

    # loss.backward()
    loss.backward(retain_graph=True)
    loss_groundtruth.backward()
    
    optimizer.step()
    scheduler.step()

    # print(i, loss.item())
    print(i, loss.item(), loss_groundtruth.item())
    model.eval()

    ## noise generation
    # noise = x.clone()
    # noise = coef * learning_rate * torch.norm(noise) * torch.randn(x.shape, requires_grad=False, device="cuda")
    # x = x + noise

t1 = time.time()

print((t1-t0)/epochs)