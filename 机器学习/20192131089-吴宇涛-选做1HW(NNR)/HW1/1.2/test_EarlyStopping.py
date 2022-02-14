from ctypes import DllCanUnloadNow
import torch 
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader,Dataset,TensorDataset
import DLdatasets
from EarlyStopping import EarlyStopping

#设置随机化种子，使实验可复现
torch.manual_seed(6)

#准备实验数据
#flux，label中每一行分别是一个样本的数据和标签
flux,label=DLdatasets.loaddatalamostspectrum(paraindex=2)


#数据维度和样本量
n_feature=flux.shape[1]
n_sample=flux.shape[0]

#训练数据，验证数据和测试数据的数量
train_size=int(n_sample*0.7)
validate_size=int(n_sample*0.1)
test_size=n_sample-validate_size-train_size

train_flux,train_label,validate_flux,validate_label,test_flux,test_label=DLdatasets.random_split(flux,label,[train_size,validate_size,test_size],dim=0)

train_flux,mean_tr,var_tr=DLdatasets.normalize_2d(train_flux)
validate_flux=DLdatasets.normalize_2d_e(validate_flux,mean_tr,var_tr)
test_flux=DLdatasets.normalize_2d_e(test_flux,mean_tr,var_tr)

train_dataset=TensorDataset(train_flux,train_label)
train_dataset=DLdatasets.mydataset(train_dataset)

validate_dataset=TensorDataset(validate_flux,validate_label)
validate_dataset=DLdatasets.mydataset(validate_dataset)

test_dataset=TensorDataset(test_flux,test_label)
test_dataset=DLdatasets.mydataset(test_dataset)

train_loader=DataLoader(train_dataset,batch_size=64,shuffle=False,num_workers=0)

validate_loader=DataLoader(validate_dataset,batch_size=64,shuffle=False,num_workers=0)

test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False,num_workers=0)

#定义模型
class NNH2Regression(torch.nn.Module):
    def __init__(self,n_feature,n_hidden1,n_hidden2,n_output):
        super(NNH2Regression,self).__init__()
        self.hidden1=torch.nn.Linear(n_feature,n_hidden1)
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

#创建模型
n_hidden1=50
n_hidden2=10
n_output=1
model=NNH2Regression(n_feature,n_hidden1,n_hidden2,n_output)

#模型学习
#创建模型优化器
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
criterion=torch.nn.MSELoss()

patience=20
early_stopping =EarlyStopping(patience,verbose=True)

def train_model(model, batch_size, patience, n_epochs):
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        model.train() # prep model for training
        for batch, (data, target) in enumerate(train_loader, 1):
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        for data, target in validate_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(n_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('1.2/checkpoint.pt'))

    return  model, avg_train_losses, avg_valid_losses

batch_size = 64
n_epochs = 300


# early stopping patience; how long to wait after last time validation loss improved.
patience = 20

model, train_loss, valid_loss = train_model(model, batch_size, patience, n_epochs)

# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

# find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss))+1 
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 0.5) # consistent scale
plt.xlim(0, len(train_loss)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('1.2/loss_plot.png', bbox_inches='tight')