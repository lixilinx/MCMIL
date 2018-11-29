#import random
import torch
from torch.autograd import grad
import torch.nn.functional as F
from torchvision import datasets, transforms
import preconditioned_stochastic_gradient_descent as psgd 
import utilities as U

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,           
                       transform=transforms.Compose([ 
                               transforms.RandomCrop(32, padding=4),
                               transforms.RandomHorizontalFlip(), 
                               transforms.ToTensor()])),    
                        batch_size=32, shuffle=True)#too slow with larger batch size on my machine
test_loader = torch.utils.data.DataLoader(    
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()])),    
                        batch_size=32, shuffle=True)
"""Test error rates of a few runs: 8.5%, 8.85%, 8.86%"""
dim1, dim2, dim3 = 128, 192, 256
W1 = torch.tensor(torch.randn(3*3*3+1, dim1)/(3*3*3)**0.5, requires_grad=True, device=device)
W2 = torch.tensor(torch.randn(dim1*3*3+1, dim1)/(dim1*3*3)**0.5, requires_grad=True, device=device)
W3 = torch.tensor(torch.randn(dim1*3*3+1, dim1)/(dim1*3*3)**0.5, requires_grad=True, device=device)
# decimation, i.e., stride=2
W4 = torch.tensor(torch.randn(dim1*3*3+1, dim2)/(dim1*3*3)**0.5, requires_grad=True, device=device)
W5 = torch.tensor(torch.randn(dim2*3*3+1, dim2)/(dim2*3*3)**0.5, requires_grad=True, device=device)
W6 = torch.tensor(torch.randn(dim2*3*3+1, dim2)/(dim2*3*3)**0.5, requires_grad=True, device=device)
# decimation, i.e., stride=2
W7 = torch.tensor(torch.randn(dim2*3*3+1, dim3)/(dim2*3*3)**0.5, requires_grad=True, device=device)
W8 = torch.tensor(torch.randn(dim3*3*3+1, dim3)/(dim3*3*3)**0.5, requires_grad=True, device=device)
W9 = torch.tensor(torch.randn(dim3*3*3+1, dim3)/(dim3*3*3)**0.5, requires_grad=True, device=device)
# detection layer
W10 = torch.tensor(torch.randn(dim3*3*3+1, 10+1)/(dim3*3*3)**0.5, requires_grad=True, device=device)
# pure CNN model
def model(x): 
    x = F.leaky_relu(F.conv2d(x, W1[:-1].view(dim1,3,3,3), bias=W1[-1], padding=1), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W2[:-1].view(dim1,dim1,3,3), bias=W2[-1], padding=1), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W3[:-1].view(dim1,dim1,3,3), bias=W3[-1], padding=1), negative_slope=0.1)
    #print(x.shape)
    x = F.leaky_relu(F.conv2d(x, W4[:-1].view(dim2,dim1,3,3), bias=W4[-1], padding=1, stride=2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W5[:-1].view(dim2,dim2,3,3), bias=W5[-1], padding=1), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W6[:-1].view(dim2,dim2,3,3), bias=W6[-1], padding=1), negative_slope=0.1)
    #print(x.shape)
    x = F.leaky_relu(F.conv2d(x, W7[:-1].view(dim3,dim2,3,3), bias=W7[-1], padding=1, stride=2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W8[:-1].view(dim3,dim3,3,3), bias=W8[-1], padding=1), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W9[:-1].view(dim3,dim3,3,3), bias=W9[-1]), negative_slope=0.1)
    #print(x.shape)
    x = F.conv2d(x, W10[:-1].view(11,dim3,3,3), bias=W10[-1])
    #print(x.shape)
    return x

def train_loss(images, labels):
    y = model(images)
    y = F.log_softmax(y, 1).double()
    loss = -torch.sum(U.log_prb_1l_per_smpl(y, labels))        
    return loss/y.shape[0]/y.shape[2]/y.shape[3]

def test_loss_approx( ):
    # detection with approximate probability 
    num_errs = 0
    with torch.no_grad():
        for data, target in test_loader:
            y = model(data.to(device))
            y = F.log_softmax(y, 1)
            y = y[:,:-1]#remove NULL class
            y = torch.exp(y)#this is the likelihood        
            y = torch.sum(y, dim=[2,3])#accumulate likelihood                
            _, pred = torch.max(y, dim=1)#make a traditional decision 
            num_errs += torch.sum(pred!=target.to(device))           
    return num_errs.item()/len(test_loader.dataset)

def test_loss_exact( ):
    # detection with exact probability
    num_errs = 0
    with torch.no_grad():
        for data, target in test_loader:
            y = model(data.to(device))
            y = F.log_softmax(y, 1)
            y = torch.exp(y)
            y = y[:,:-1] + y[:,-1:]
            y = torch.sum(torch.log(y), dim=[2,3])
            _, pred = torch.max(y, dim=1)#make a traditional decision 
            num_errs += torch.sum(pred!=target.to(device))           
    return num_errs.item()/len(test_loader.dataset)
    
# train and test our model; use PSGD-Newton for optimization (virtually tuning free)
Ws = [W1,W2,W3,W4,W5,W6,W7,W8,W9,W10]
Qs = [[torch.eye(W.shape[0], device=device), torch.eye(W.shape[1], device=device)] for W in Ws]
step_size = 0.01
num_epochs = 64
grad_norm_clip_thr = 0.1*sum(W.shape[0]*W.shape[1] for W in Ws)**0.5 
TrainLoss, TestLossApprox, TestLossExact = [], [], []
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        #new_size = random.randint(28, 36)#random height rescaling
        #data = data[:,:,(torch.arange(new_size)*(32-1)/(new_size-1)).long()]
        #new_size = random.randint(28, 36)#random width rescaling
        #data = data[:,:,:,(torch.arange(new_size)*(32-1)/(new_size-1)).long()]
    
        loss = train_loss(data.to(device), target.to(device))

        grads = grad(loss, Ws, create_graph=True)
        TrainLoss.append(loss.item())
        if batch_idx%100==0:
            print('Epoch: {}; batch: {}; train loss: {}'.format(epoch, batch_idx, TrainLoss[-1]))          

        v = [torch.randn(W.shape, device=device) for W in Ws]
        Hv = grad(grads, Ws, v)#just let Hv=grads if using whitened gradients   
        with torch.no_grad():
            Qs = [psgd.update_precond_kron(q[0], q[1], dw, dg) for (q, dw, dg) in zip(Qs, v, Hv)]
            pre_grads = [psgd.precond_grad_kron(q[0], q[1], g) for (q, g) in zip(Qs, grads)]
            grad_norm = torch.sqrt(sum([torch.sum(g*g) for g in pre_grads]))
            step_adjust = min(grad_norm_clip_thr/(grad_norm + 1.2e-38), 1.0)
            for i in range(len(Ws)):
                Ws[i] -= step_adjust*step_size*pre_grads[i]
                
    TestLossApprox.append(test_loss_approx())
    TestLossExact.append(test_loss_exact())
    print('Epoch: {}; best test loss (approximate): {}; best test loss (exact): {}'.format(epoch, min(TestLossApprox), min(TestLossExact)))
            
    if epoch+1 == int(num_epochs/2):
        step_size *= 0.1