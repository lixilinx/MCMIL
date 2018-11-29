import torch
from torch.autograd import grad
import torch.nn.functional as F
from torchvision import datasets, transforms
import preconditioned_stochastic_gradient_descent as psgd 
import utilities as U

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,           
                       transform=transforms.Compose([                       
                               transforms.ToTensor()])),    
                        batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(    
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()])),    
                        batch_size=64, shuffle=True)

"""Test error rates from a few runs: 0.41%, 0.51%, 0.52%, 0.5%, 0.46%"""
W1 = torch.tensor(torch.randn(1*5*5+1, 64)/(1*5*5)**0.5, requires_grad=True, device=device)
W2 = torch.tensor(torch.randn(64*5*5+1, 64)/(64*5*5)**0.5, requires_grad=True, device=device)
W3 = torch.tensor(torch.randn(64*5*5+1, 64)/(64*5*5)**0.5, requires_grad=True, device=device)
W4 = torch.tensor(torch.randn(64*5*5+1, 64)/(64*5*5)**0.5, requires_grad=True, device=device)
W5 = torch.tensor(torch.randn(64*5*5+1, 64)/(64*5*5)**0.5, requires_grad=True, device=device)
W6 = torch.tensor(torch.randn(64*5*5+1, 10+1)/(64*5*5)**0.5, requires_grad=True, device=device)
def model(x): 
    x = F.leaky_relu(F.conv2d(x, W1[:-1].view(64,1,5,5), bias=W1[-1], padding=2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W2[:-1].view(64,64,5,5), bias=W2[-1], padding=2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W3[:-1].view(64,64,5,5), bias=W3[-1], padding=2, stride=2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W4[:-1].view(64,64,5,5), bias=W4[-1], padding=1), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W5[:-1].view(64,64,5,5), bias=W5[-1]), negative_slope=0.1)
    x = F.conv2d(x, W6[:-1].view(11,64,5,5), bias=W6[-1])
    #print(x.shape)
    return x

def train_loss(images, labels):
    y = model(images)
    y = F.log_softmax(y, 1)
    loss = -torch.sum(U.log_prb_1l_per_smpl(y, labels))      
    return loss/y.shape[0]/y.shape[2]/y.shape[3]

def test_loss_approx( ):
    # detection with approximated probability 
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
Ws = [W1,W2,W3,W4,W5,W6]
Qs = [[torch.eye(W.shape[0], device=device), torch.eye(W.shape[1], device=device)] for W in Ws]
step_size = 0.02
num_epochs = 20
grad_norm_clip_thr = 0.1*sum(W.shape[0]*W.shape[1] for W in Ws)**0.5 
TrainLoss, TestLossApprox, TestLossExact = [], [], []
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
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