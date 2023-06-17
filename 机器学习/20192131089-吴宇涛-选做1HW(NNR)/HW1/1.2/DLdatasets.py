import torch 
import numpy as np 
from torch.utils.data import Dataset


def linesine():
    w0=1.2
    b0=0.4
    w1=2
    sigma=1.4

    #准备训练数据，每一行是一个样本
    x_tr=torch.unsqueeze(torch.linspace(-6,6,200),dim=1)
    y_ref_tr=w0*x_tr+b0+w1*x_tr.sin()
    y_tr=y_ref_tr+sigma*torch.randn(x_tr.size())

    #准备测试数据
    x_te=12*torch.rand(180)-6
    x_te=x_te.unsqueeze(1)
    y_ref_te=w0*x_te+b0+w1*x_te.sin()
    y_te=y_ref_te+sigma*torch.randn(x_te.size())

    #准备验证数据
    x_val=12*torch.rand(100)-6
    x_val=x_val.unsqueeze(1)
    y_ref_val=w0*x_val+b0+w1*x_val.sin()
    y_val=y_ref_val+sigma*torch.randn(x_val.size())

    return x_tr,y_ref_tr,y_tr,x_te,y_te,x_val,y_val


def loaddatalamostspectrum(paraindex=2):
    flux=np.load('LAMOST_APOGEE/flux_end_train_10_20.npy')
    label=np.load('LAMOST_APOGEE/label_ap_la_10_20.npy')
    n_feature=flux.shape[1]
    flux=torch.tensor(flux,dtype=torch.float32)
    label=torch.tensor(label[:,paraindex],dtype=torch.float32)
    label=label.unsqueeze(1)

    return flux,label
def random_split(x,y,sizes,dim=0):
    np.random.seed(6)
    index1=np.arange(x.shape[0])
    np.random.shuffle(index1)
    if dim==0:
        if len(sizes)==2:
            x1=x[index1[0:sizes[0]],:]
            y1=y[index1[0:sizes[0]],:]
            x2=x[index1[sizes[0]:],:]
            y2=y[index1[sizes[0]:],:]
            return x1,y1,x2,y2
        else :
            x1=x[index1[0:sizes[0]],:]
            y1=y[index1[0:sizes[0]],:]
            x2=x[index1[sizes[0]:sizes[0]+sizes[1]],:]
            y2=y[index1[sizes[0]:sizes[0]+sizes[1]],:]
            x3=x[index1[sizes[0]+sizes[1]:],:]
            y3=y[index1[sizes[0]+sizes[1]:],:]
            return x1,y1,x2,y2,x3,y3    

def normalize_2d_e(x,mean,var):
    x=(x-mean)/var
    return x


def normalize_2d(x):
    var=torch.var(x,dim=0)
    mean=torch.mean(x,dim=0)
    x=(x-mean)/var
    return x,mean,var
    

def normalize_2d_inverse(x,mean=0,var=1):
    x=x*var+mean
    
class mydataset(Dataset):
    def __init__(self,loaded_data):
        self.data=loaded_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]





 