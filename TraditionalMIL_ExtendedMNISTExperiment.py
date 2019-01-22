import random
import pickle
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
"""The same settings as extended_MNIST_experiment. 
The traditional MIL cost is difficult to optimize when the model is highly nonconvex"""
"""Test error rates from a few runs: 8.9%, 5.4%, 8.4%"""
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
    y = y - torch.max(y)#prevent overflow
    y = torch.exp(y)
    y = torch.log(y/torch.sum(y))
    loss = 0.0
    for i in range(y.shape[0]):
        for l in labels[i]:
            loss -= torch.max(y[i,l])
    return loss/y.shape[0]

def test_loss( ):
    num_errs = 0
    with torch.no_grad():
        for data, target in test_loader:
            y = model(data.to(device))
            y, _ = torch.max(y, dim=3)
            y, _ = torch.max(y, dim=2)
            _, pred = torch.max(y, dim=1)
            num_errs += torch.sum(pred!=target.to(device))
            
    return num_errs.item()/len(test_loader.dataset)

# train and test our model; use PSGD-Newton for optimization (virtually tuning free)
Ws = [W1,W2,W3,W4,W5,W6]
Qs = [[torch.eye(W.shape[0], device=device), torch.eye(W.shape[1], device=device)] for W in Ws]
step_size = 0.02
num_epochs = 20
grad_norm_clip_thr = 0.1*sum(W.shape[0]*W.shape[1] for W in Ws)**0.5 
TrainLoss, TestLoss, BestTestLoss = [], [], 1e30
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        nestH, nestW = random.randint(56, 84), random.randint(56, 84)
        new_data = torch.zeros(int(data.shape[0]/2), data.shape[1], nestH, nestW)
        new_target = []
        for i in range(int(data.shape[0]/2)):
            new_data[i] = U.nest_images(data[2*i:2*i+2], nestH, nestW)
            new_target.append(U.remove_repetitive_labels([target[2*i].item(), target[2*i+1].item()]))#just a list of int
        
        loss = train_loss(new_data.to(device), new_target)

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
                
    TestLoss.append(test_loss())
    print('Epoch: {}; best test loss: {}'.format(epoch, min(TestLoss)))
    if TestLoss[-1] < BestTestLoss:
        BestTestLoss = TestLoss[-1]
        with open('mnist_model', 'wb') as f:
            pickle.dump(Ws, f)
    if epoch+1 == int(num_epochs/2):
        step_size = 0.1*step_size