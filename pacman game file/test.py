import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# #implementation of framework of target net and evaluation net
# class Net(nn.Module):
#     def __init__(self, action_num, feature_num):
#         super(Net, self).__init__()
#         #the first fully connect layer
#         self.fc1 = nn.Linear(feature_num, 20, bias=False)
#         #randomly initialize the fc layer
#         self.fc1.weight.data.normal_(0,0.1)
#         #drop out layer 1
#         self.drop1 = nn.Dropout(0.2)
#         # the second fully connect layer
#         self.fc2 = nn.Linear(20, 10, bias=False)
#         # randomly initialize the fc layer
#         self.fc2.weight.data.normal_(0, 0.1)
#         # drop out layer 1
#         self.drop1 = nn.Dropout(0.1)
#         #out_layer
#         self.out = nn.Linear(10, action_num, bias=False)
#         self.out.weight.data.normal_(0,0.1)
#         #forward function in pytorch module
#     def forward(self,x):
#         #first layer
#         x = self.fc1(x)
#         #x = self.drop1(x)
#         x = F.leaky_relu(x,0.01)
#         #second layer
#         x = self.fc2(x)
#         #x = self.drop1(x)
#         x = F.leaky_relu(x,0.01)
#         #output layer
#         actions_value = self.out(x)
#         return actions_value
# eval_net = Net(4,10)
# target_net = Net(4,10)
# # x = range(10)
# # x = torch.unsqueeze(torch.FloatTensor(x), 0)
# # print(x)
#
# x = np.zeros((3,10))
# y = 3
# y = torch.FloatTensor(y)
# x = torch.FloatTensor(x)
# print(x)
# print(y.shape)
# m1 = eval_net.forward(x)
# print(m1)
# print((y*m1.max(1)[0]).shape)
print(([5,3]==[5,3])and(1==1))