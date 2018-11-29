import random
import pickle
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
import torch.nn.functional as F
import preconditioned_stochastic_gradient_descent as psgd 
import utilities as U

batch_size = 32
lr = 0.01 
max_num_iter = int(1e5)#1e5 iterations are about 13.6 epochs; totally 235755 samples
model_name = 'svhn_model'
enable_dropout = False#our model is small (compared with others for this task), dropout is not very helpful
train_from_sketch = True
pre_check_svhn_reading = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
# I prepared these mat files with matlab/octave; see readme
train_images, train_labels = U.read_svhn_mat('./svhn/train.mat')
len_train = len(train_images)
rp = np.random.permutation(len_train)
train_images = train_images[rp]
train_labels = train_labels[rp]

extra1_images, extra1_labels = U.read_svhn_mat('./svhn/extra_part1.mat')
len_extra1 = len(extra1_images)
rp = np.random.permutation(len_extra1)
extra1_images = extra1_images[rp]
extra1_labels = extra1_labels[rp]

extra2_images, extra2_labels = U.read_svhn_mat('./svhn/extra_part2.mat')
len_extra2 = len(extra2_images)
rp = np.random.permutation(len_extra2)
extra2_images = extra2_images[rp]
extra2_labels = extra2_labels[rp]

def get_batch():#read training data; do some augumentations
    choice = random.randint(0,2)
    if choice==0:
        i = random.randint(0, len_train - batch_size)
        images = train_images[i:i+batch_size]
        labels = train_labels[i:i+batch_size]
    elif choice==1:
        i = random.randint(0, len_extra1 - batch_size)
        images = extra1_images[i:i+batch_size]
        labels = extra1_labels[i:i+batch_size] 
    else:
        i = random.randint(0, len_extra2 - batch_size)
        images = extra2_images[i:i+batch_size]
        labels = extra2_labels[i:i+batch_size]    
        
    rand_shift1, rand_shift2 = random.randint(0, 4), random.randint(0, 4)
    images = images[:,:, rand_shift1:rand_shift1+60, rand_shift2:rand_shift2+60]#with shape (B, C, H, W)
    images = images[:, np.random.permutation(3)]#randomly permute (R, G, B)
    #if np.random.randn()>0:#random RGB gains
    #    images = images*np.random.rand(1,3,1,1)
    old_height, old_width = images.shape[2], images.shape[3]
    new_size = random.randint(40, 80)#random height rescaling
    images = images[:,:,(np.arange(new_size)*(old_height-1)/(new_size-1)).astype(int)]
    new_size = random.randint(40, 80)#random width rescaling
    images = images[:,:,:,(np.arange(new_size)*(old_width-1)/(new_size-1)).astype(int)]
    return images, labels

if pre_check_svhn_reading:#do the data make sense?
    for _ in range(10):
        images, labels = get_batch()
        plt.imshow(np.transpose(images[0], [1,2,0])/256)
        plt.title(str(labels[0]))
        plt.show()
    time.sleep(1)
   
if train_from_sketch:
    # model coefficients
    ks, dim1, dim2, dim3 = 5, 48, 64, 96
    dim0 = 3#RGB images
    W1 = torch.tensor(torch.randn(dim0*ks*ks+1, dim1)/(dim0*ks*ks)**0.5, requires_grad=True, device=device)
    W2 = torch.tensor(torch.randn(dim1*ks*ks+1, dim1)/(dim1*ks*ks)**0.5, requires_grad=True, device=device)
    W3 = torch.tensor(torch.randn(dim1*ks*ks+1, dim1)/(dim1*ks*ks)**0.5, requires_grad=True, device=device)
    # decimation by 2, i.e., stride=2
    W4 = torch.tensor(torch.randn(dim1*ks*ks+1, dim2)/(dim1*ks*ks)**0.5, requires_grad=True, device=device)
    W5 = torch.tensor(torch.randn(dim2*ks*ks+1, dim2)/(dim2*ks*ks)**0.5, requires_grad=True, device=device)
    W6 = torch.tensor(torch.randn(dim2*ks*ks+1, dim2)/(dim2*ks*ks)**0.5, requires_grad=True, device=device)
    # another three layers
    W7 = torch.tensor(torch.randn(dim2*ks*ks+1, dim3)/(dim2*ks*ks)**0.5, requires_grad=True, device=device)
    W8 = torch.tensor(torch.randn(dim3*ks*ks+1, dim3)/(dim3*ks*ks)**0.5, requires_grad=True, device=device)
    W9 = torch.tensor(torch.randn(dim3*ks*ks+1, dim3)/(dim3*ks*ks)**0.5, requires_grad=True, device=device)
    # detection layer
    W10 =torch.tensor(torch.randn(dim3*ks*ks+1, 10+1)/(dim3*ks*ks)**0.5, requires_grad=True, device=device)
else:
    with open(model_name, 'rb') as f:
        Ws = pickle.load(f)
        W1,W2,W3,W4,W5,W6,W7,W8,W9,W10 = Ws
        ks, dim1, dim2, dim3 = int((W3.shape[0]/W3.shape[1])**0.5), W3.shape[1], W6.shape[1], W9.shape[1]
        dim0 = 3#RGB images
# CNN model
def model(x): 
    x = F.leaky_relu(F.conv2d(x, W1[:-1].view(dim1,dim0,ks,ks), bias=W1[-1], padding=ks//2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W2[:-1].view(dim1,dim1,ks,ks), bias=W2[-1], padding=ks//2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W3[:-1].view(dim1,dim1,ks,ks), bias=W3[-1], padding=ks//2), negative_slope=0.1)
    if enable_dropout:
        x = 2*torch.bernoulli(torch.rand(x.shape,device=device))*x  
    #print(x.shape)
    x = F.leaky_relu(F.conv2d(x, W4[:-1].view(dim2,dim1,ks,ks), bias=W4[-1], padding=ks//2, stride=2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W5[:-1].view(dim2,dim2,ks,ks), bias=W5[-1], padding=ks//2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W6[:-1].view(dim2,dim2,ks,ks), bias=W6[-1], padding=ks//2), negative_slope=0.1)
    if enable_dropout:
        x = 2*torch.bernoulli(torch.rand(x.shape,device=device))*x
    #print(x.shape)   
    x = F.leaky_relu(F.conv2d(x, W7[:-1].view(dim3,dim2,ks,ks), bias=W7[-1], padding=ks//2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W8[:-1].view(dim3,dim3,ks,ks), bias=W8[-1], padding=ks//2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W9[:-1].view(dim3,dim3,ks,ks), bias=W9[-1], padding=ks//2), negative_slope=0.1)
    if enable_dropout:
        x = 2*torch.bernoulli(torch.rand(x.shape,device=device))*x
    #print(x.shape)
    x = F.conv2d(x, W10[:-1].view(10+1,dim3,ks,ks), bias=W10[-1])
    #print(x.shape)
    return x

def train_loss(images, labels):
    y = model(images)
    y = F.log_softmax(y, 1).double()#really need double precision to calculate log Prb({lables}|Image)
    loss = 0.0
    for i in range(y.shape[0]):
        loss -= U.log_prb_labels(y[i], labels[i])    
    return loss/y.shape[0]/y.shape[2]/y.shape[3]

# train our model; use PSGD-Newton for optimization (virtually tuning free)
Ws = [W1,W2,W3,W4,W5,W6,W7,W8,W9,W10]
Qs = [[torch.eye(W.shape[0], device=device), torch.eye(W.shape[1], device=device)] for W in Ws]
grad_norm_clip_thr = 0.1*sum(W.shape[0]*W.shape[1] for W in Ws)**0.5 
TrainLoss = []
t0 = time.time()
for num_iter in range(max_num_iter):
    images, labels = get_batch()
        
    new_images = torch.tensor(images/256, dtype=torch.float, device=device)
    new_labels = []
    for label in labels:
        new_labels.append(U.remove_repetitive_labels(label))
        
    loss = train_loss(new_images, new_labels)
    grads = grad(loss, Ws, create_graph=True)
    TrainLoss.append(loss.item())     

    v = [torch.randn(W.shape, device=device) for W in Ws]
    Hv = grad(grads, Ws, v)  
    with torch.no_grad():
        Qs = [psgd.update_precond_kron(q[0], q[1], dw, dg) for (q, dw, dg) in zip(Qs, v, Hv)]
        pre_grads = [psgd.precond_grad_kron(q[0], q[1], g) for (q, g) in zip(Qs, grads)]
        grad_norm = torch.sqrt(sum([torch.sum(g*g) for g in pre_grads]))
        lr_adjust = min(grad_norm_clip_thr/(grad_norm + 1.2e-38), 1.0)
        for i in range(len(Ws)):
            Ws[i] -= lr_adjust*lr*pre_grads[i]
        
    if num_iter%100==0:     
        print('iter: {}; train loss: {}; elapsed time: {}'.format(num_iter, TrainLoss[-1], time.time()-t0))
        with open(model_name, 'wb') as f:
            pickle.dump(Ws, f)
            
    if num_iter==int(max_num_iter/2):
        lr *= 0.1