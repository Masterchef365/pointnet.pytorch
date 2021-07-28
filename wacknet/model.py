from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from pointnet.model import PointNetfeat

class WackNet(nn.Module):
    def __init__(self, narrow_width, wide_width, n_rows, k = 2, feature_transform=False):
        super(WackNet, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        conv1_inp_size = 1024 + 64 + 10*4 # Features, point features, and all combined image classes
        self.conv1 = torch.nn.Conv1d(conv1_inp_size, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.top_convnet = ConvNet(wide_width, n_rows)
        self.right_convnet = ConvNet(narrow_width, n_rows)
        self.bottom_convnet = ConvNet(wide_width, n_rows)
        self.left_convnet = ConvNet(narrow_width, n_rows)

    def forward(self, pcld, top, right, bottom, left):
        batchsize = pcld.size()[0]
        n_pts = pcld.size()[2]

        top_cls = self.top_convnet(top)
        right_cls = self.right_convnet(right)
        bottom_cls = self.bottom_convnet(bottom)
        left_cls = self.left_convnet(left)
        combined = torch.cat([top_cls, right_cls, bottom_cls, left_cls], dim=1)
        tiled = combined.unsqueeze(2).repeat(1, 1, n_pts)

        x, trans, trans_feat = self.feat(pcld)

        x = torch.cat([x, tiled], dim=1) # Append to the features for each point...

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

class ConvNet(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Like, I literally... I don't even know dude
        width //= 2; height //= 2
        width -= 2; height -= 2
        width //= 2; height //= 2
        width -= 2; height -= 2
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * width * height, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x