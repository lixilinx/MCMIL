import pickle
import random
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import utilities as U

test_loader = torch.utils.data.DataLoader(    
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()])),    
                        batch_size=10, shuffle=True)
    
#with open('mnist_model', 'rb') as f:#
with open('mnist_model_test_error_rate_0p0032', 'rb') as f:
    Ws = pickle.load(f)
    W1,W2,W3,W4,W5,W6 = Ws 
    device = W1.device
    
def model(x): 
    x = F.leaky_relu(F.conv2d(x, W1[:-1].view(64,1,5,5), bias=W1[-1], padding=2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W2[:-1].view(64,64,5,5), bias=W2[-1], padding=2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W3[:-1].view(64,64,5,5), bias=W3[-1], padding=2, stride=2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W4[:-1].view(64,64,5,5), bias=W4[-1], padding=1), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W5[:-1].view(64,64,5,5), bias=W5[-1]), negative_slope=0.1)
    x = F.conv2d(x, W6[:-1].view(11,64,5,5), bias=W6[-1])
    return x

plt_cnt = 0
plt.figure()
for batch_idx, (data, target) in enumerate(test_loader):
    random_size = random.randint(128, 192)
    new_data = U.nest_images(data, random_size, random_size)
    y = model(new_data[None,:,:,:].to(device))[0]

    plt_cnt += 1
    if plt_cnt<3:
        plt.subplot(2,2,2*plt_cnt)
    else:
        break
    
    for i in range(y.shape[1]):
        for j in range(y.shape[2]):
            _, label = torch.max(y[:,i,j], dim=0)
            if label < 10:
                plt.text(j/y.shape[2], 1-i/y.shape[1], str(label.item()))
    
    if plt_cnt==1:            
        plt.title('recognition results')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    
    plt.subplot(2,2,2*plt_cnt-1)
    plt.imshow(new_data[0])
    if plt_cnt==1:
        plt.title('input images')
    plt.axis('off')
    
#plt.savefig('test.eps', dpi=150)
plt.show()