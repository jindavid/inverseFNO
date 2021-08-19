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

# class SimpleBlock1d_layer(nn.Module):
#     def __init__(self, modes, width, layer):
#         super(SimpleBlock1d_layer, self).__init__()
#
#         self.modes1 = modes
#         self.width = width
#         self.layer = layer
#         self.fc0 = nn.Linear(2, self.width)
#
#         self.conv_list = []
#         self.w_list = []
#         self.bn_list = []
#
#         for l in range(self.layer):
#             self.conv_list.append(SpectralConv1d_fast(self.width, self.width, self.modes1))
#             self.w_list.append(nn.Conv1d(self.width, self.width, 1))
#             self.bn_list.append(torch.nn.BatchNorm1d(self.width))
#
#         self.conv_list = nn.ModuleList(self.conv_list)
#         self.w_list = nn.ModuleList(self.w_list)
#         self.bn_list = nn.ModuleList(self.bn_list)
#
#
#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, 1)
#
#     def forward(self, x):
#         batchsize = x.shape[0]
#         size_x = x.shape[1]
#
#         x = self.fc0(x)
#         x = x.permute(0, 2, 1)
#
#         for l in range(self.layer):
#             x1 = self.conv_list[l](x)
#             x2 = self.w_list[l](x)
#             x = x1 + x2
#             # x = self.bn_list[l](x1 + x2)
#             x = F.relu(x)
#
#         x = x.permute(0, 2, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         return x
#
# class Net1d_layer(nn.Module):
#     def __init__(self, modes, width, layer):
#         super(Net1d_layer, self).__init__()
#
#         self.conv1 = SimpleBlock1d_layer(modes, width, layer)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x.squeeze()
#
#     def count_params(self):
#         c = 0
#         for p in self.parameters():
#             c += reduce(operator.mul, list(p.size()))
#
#         return c

# total_index = 0
# total_error = np.zeros((5,500,4))
# for width in [1024, 256, 64, 16, 4]:
#     print(total_index)

ntrain = 1000
ntest = 200

sub = 8 #subsampling rate
h = 2**13 // sub
s = h

batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 100
gamma = 0.5

modes = 16
width = 64

path = 'burgers_fourier_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width) #+ '_l' + str(layer)
path_model = 'model/'+path
path_pred = 'pred/'+path+'.mat'
path_image = 'image/'+path

# dataloader = MatReader('data/burgers_v1000_N1200.mat')
dataloader = MatReader('data/burgers_data_R10.mat')
x_data = dataloader.read_field('a')[:,::sub]
y_data = dataloader.read_field('u')[:,::sub]

x_train = x_data[:ntrain,:]
y_train = y_data[:ntrain,:]
x_test = x_data[-ntest:,:]
y_test = y_data[-ntest:,:]

# x_normalizer = UnitGaussianNormalizer(x_train)
# x_train = x_normalizer.encode(x_train)
# x_test = x_normalizer.encode(x_test)
#
# y_normalizer = UnitGaussianNormalizer(y_train)
# y_train = y_normalizer.encode(y_train)

grid = np.linspace(0, 1, s).reshape(1, s, 1)
grid = torch.tensor(grid, dtype=torch.float)
x_train = torch.cat([x_train.reshape(ntrain,s,1), grid.repeat(ntrain,1,1)], dim=2)
x_test = torch.cat([x_test.reshape(ntest,s,1), grid.repeat(ntest,1,1)], dim=2)


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# model = Net1d_old().cuda()
model = Net1d(modes, width).cuda()
# model = Net1d_layer(modes, width, layer).cuda()

num_param = model.count_params()
print(num_param)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
error = np.zeros((epochs, 4))
# y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)

        mse = F.mse_loss(out, y, reduction='mean')
        # mse.backward()

        # y = y_normalizer.decode(y)
        # out = y_normalizer.decode(out)
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    test_mse = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            # out = y_normalizer.decode(out)
            test_mse += F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean').item()
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse /= len(train_loader)
    test_mse /= len(test_loader)
    train_l2 /= ntrain
    test_l2 /= ntest

    error[ep] = [train_mse, test_mse, train_l2, test_l2]

    t2 = default_timer()
    print(ep, t2-t1, train_mse, test_mse, train_l2, test_l2)

torch.save(model, 'model/ns_fourier_burgers_8192')
pred = torch.zeros(y_test.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.cuda(), y.cuda()

        out = model(x)
        # out = y_normalizer.decode(out)
        pred[index] = out

        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        print(index, test_l2)
        index = index + 1

# scipy.io.savemat(path_pred, mdict={'pred': pred.cpu().numpy(), 'error': error, 'num_param': num_param})


#     total_error[total_index] = error
#     total_index = total_index + 1
#
# scipy.io.savemat('burgers_fourier_full_width_updown.mat', mdict={'error': total_error})
