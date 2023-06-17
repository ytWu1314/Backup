import numpy
import torch 
import matplotlib.pyplot as plt
import DLdatasets

#设置随机化种子，使实现可复现
torch.manual_seed(6)

x_tr,y_ref_tr,y_tr,x_te,y_te,x_val,y_val=DLdatasets.linesine()

#训练数据的维度

n_feature=x_tr.shape[1]

if n_feature==1:
    fig=plt.figure(dpi=600)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x_tr.numpy(),y_tr.numpy(),'k*')
    plt.plot(x_tr.numpy(),y_ref_tr.numpy(),linestyle='--',color='k')
    plt.legend(['Observed samples','Theoretical Model'],loc='lower right')
    plt.savefig('1.1/NNRH1regression.pdf')
    plt.savefig('1.1/NNRH1regression.eps')
    plt.savefig('1.1/NNRH1regression.png')

class NNRH1regression(torch.nn.Module):
    def __init__(self,nfeature,n_hidden1,n_hidden2,n_output):
        super(NNRH1regression,self).__init__()
        self.hidden1=torch.nn.Linear(nfeature,n_hidden1)
        self.hidden2=torch.nn.Linear(n_hidden1,n_hidden2)
        self.predict=torch.nn.Linear(n_hidden2,n_output)
    def forward(self,x):
        h1z=self.hidden1(x)
        h1a=torch.tanh(h1z)
        h2z=self.hidden2(h1a)
        h2a=torch.tanh(h2z)
        nnout=self.predict(h2a)
        return nnout    
    def learning(self,x,y,loss_func,optimizer):
        prediction=self.forward(x)
        loss=loss_func(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

#两个隐藏层 分别有4个神经元和2个神经元

n_hidden1=4
n_hidden2=2
n_output=1
NNR=NNRH1regression(n_feature,n_hidden1,n_hidden2,n_output)


optimizer=torch.optim.SGD(NNR.parameters(),lr=0.01)
loss_func_mse=torch.nn.MSELoss()

n_epochs=20000
loss_tensor=torch.zeros(n_epochs)
for epoch in range(1,n_epochs+1):
    loss=NNR.learning(x_tr,y_tr,loss_func_mse,optimizer)
    loss_tensor[epoch-1]=loss

pred_tr=NNR(x_tr)
if n_feature==1:
    fig=plt.figure(dpi=600)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x_tr.squeeze().numpy(),y_tr.squeeze().numpy(),"*")
    plt.plot(x_tr.squeeze().numpy(),pred_tr.detach().squeeze().numpy(),ls='-',color='k')
    plt.plot(x_tr.squeeze().numpy(),y_ref_tr.squeeze().numpy(),ls='--',color='b')
    plt.legend(['Observered samples','Estimation Model','Theoretical Model'],loc='lower right',prop={'size':8})
    plt.savefig('1.1/NNRH1regressionEstimation.pdf')
    plt.savefig('1.1/NNRH1regressionEstimation.eps')
    plt.savefig('1.1/NNRH1regressionEstimation.png')

fig=plt.figure(dpi=600)
plt.xlabel('epochs')
plt.ylabel('mse')
plt.plot(range(1,n_epochs+1),loss_tensor.detach().numpy())
plt.savefig('1.1/NNRH1regressionMSE.pdf')
plt.savefig('1.1/NNRH1regressionMSE.eps')
plt.savefig('1.1/NNRH1regressionMSE.png')

print('数据的理论噪声水平（均方差）:1.4')
with torch.no_grad():
    y_pred_tr=NNR(x_tr)
    lossmse_tr=loss_func_mse(y_pred_tr,y_tr)
    print('训练均方误差:',lossmse_tr.numpy(),'训练均方误差的平方根',lossmse_tr.sqrt().numpy())
with torch.no_grad():
    y_pred_te=NNR(x_te)
    lossmse_te=loss_func_mse(y_pred_te,y_te)
    print('训练均方误差:',lossmse_te.numpy(),'训练均方误差的平方根',lossmse_te.sqrt().numpy())    