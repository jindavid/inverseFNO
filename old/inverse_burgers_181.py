import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *
import random
import time

torch.manual_seed(0)
np.random.seed(0)

#Complex multiplication
def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", a, b)

    # op = partial(torch.einsum, "bix,iox->box")
    # return torch.stack([
    #     op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
    #     op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    # ], dim=-1)

################################################################
#  1d fourier layer
################################################################

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1


        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-1)), dim=[2])
        return x

class SimpleBlock1d(nn.Module):
    def __init__(self, modes, width):
        super(SimpleBlock1d, self).__init__()

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm1d(self.width)
        self.bn1 = torch.nn.BatchNorm1d(self.width)
        self.bn2 = torch.nn.BatchNorm1d(self.width)
        self.bn3 = torch.nn.BatchNorm1d(self.width)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x = x.shape[1]

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        # x = self.bn0(x1 + x2)
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        # x = self.bn1(x1 + x2)
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2

        # x = self.bn2(x1 + x2)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        # x = self.bn3(x1 + x2)


        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class Net1d(nn.Module):
    def __init__(self, modes, width):
        super(Net1d, self).__init__()
        self.conv1 = SimpleBlock1d(modes, width)


    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

################################################################
#  configurations
################################################################

ntrain = 1000

sub = 2**3 #subsampling rate
h = 2**13 // sub #total grid size divided by the subsampling rate
s = h

learning_rate = 0.1
coef = 0.0001

epochs = 1500
step_size = 100
gamma = 0.5

################################################################
# read data
################################################################

# Data is of the shape (number of samples, grid size)
dataloader = MatReader('/home2/jindawei/fourier_neural_operator/data/burgers_data_R10.mat')
x_data = dataloader.read_field('a')[:,::sub]
y_data = dataloader.read_field('u')[:,::sub]

index = random.randint(0,ntrain)
x_groundtruth = x_data[index,:]
y = y_data[index,:]

grid = np.linspace(0, 2*np.pi, s).reshape(1, s, 1)
grid = torch.tensor(grid, dtype=torch.float)
x_groundtruth = torch.cat([x_groundtruth.reshape(1,s,1), grid.repeat(1,1,1)], dim=2)
y = y.reshape(1,1024)

################################################################
# load model
################################################################

PATH = '/home2/jindawei/fourier_neural_operator/model/ns_fourier_burgers_8192'
model = torch.load(PATH)

################################################################
# Inverse
################################################################

#randomly generate parameters
x = torch.rand(x_groundtruth.shape, requires_grad=True, device="cuda")
# x = torch.norm(F.normalize(x_groundtruth, p=2, dim=1)).weight * torch.rand(x_groundtruth.shape, requires_grad=True).weight
# print(x)
# exit()
y, x_groundtruth = y.cuda(), x_groundtruth.cuda()
lr = 0.1
optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
myloss = nn.MSELoss()

t0 = time.time()

# x_first = x.clone()

for i in range(epochs):

    optimizer.zero_grad()
    output = model(x)
    loss = myloss(output.view(1, -1), y.view(1, -1))
    loss_groundtruth = myloss(x.view(1, -1), x_groundtruth.view(1, -1))

    loss.backward(retain_graph=True)
    loss_groundtruth.backward()
    
    optimizer.step()
    scheduler.step()

    print(i, loss.item(), loss_groundtruth.item())
    model.eval()


    # noise generation
    # noise = x.clone()
    # noise = coef * learning_rate * torch.norm(noise) * torch.randn(x.shape, requires_grad=False, device="cuda")
    # x = x + noise

# x_final = x.clone()

t1 = time.time()

print((t1-t0)/epochs)