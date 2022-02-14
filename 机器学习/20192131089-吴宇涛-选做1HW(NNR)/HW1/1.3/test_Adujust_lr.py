import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
 
initial_lr = 0.1
 
net_1=nn.Sequential(
    nn.Linear(1,10)
)
 
optimizer_1 = torch.optim.Adam(
    net_1.parameters(), 
    lr = initial_lr)
 
scheduler_1 = LambdaLR(
    optimizer_1, 
    lr_lambda=lambda epoch: 1/(epoch+1))
 
print("初始化的学习率：", optimizer_1.defaults['lr'])
 
for epoch in range(1, 11):
    optimizer_1.zero_grad()
    optimizer_1.step()
    print("第%d个epoch的学习率：%f" % (epoch, optimizer_1.param_groups[0]['lr']))
    scheduler_1.step()