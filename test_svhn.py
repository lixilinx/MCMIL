import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import utilities as U

#with open('svhn_model_ks_5_dims_48_64_96', 'rb') as f:
with open('svhn_model_ks_5_dims_96_128_192', 'rb') as f:
    Ws = pickle.load(f)
    W1,W2,W3,W4,W5,W6,W7,W8,W9,W10 = Ws
    ks, dim1, dim2, dim3 = int((W3.shape[0]/W3.shape[1])**0.5), W3.shape[1], W6.shape[1], W9.shape[1]
    dim0 = 3#RGB images
    device = W1.device
def model(x): 
    x = F.leaky_relu(F.conv2d(x, W1[:-1].view(dim1,dim0,ks,ks), bias=W1[-1], padding=ks//2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W2[:-1].view(dim1,dim1,ks,ks), bias=W2[-1], padding=ks//2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W3[:-1].view(dim1,dim1,ks,ks), bias=W3[-1], padding=ks//2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W4[:-1].view(dim2,dim1,ks,ks), bias=W4[-1], padding=ks//2, stride=2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W5[:-1].view(dim2,dim2,ks,ks), bias=W5[-1], padding=ks//2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W6[:-1].view(dim2,dim2,ks,ks), bias=W6[-1], padding=ks//2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W7[:-1].view(dim3,dim2,ks,ks), bias=W7[-1], padding=ks//2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W8[:-1].view(dim3,dim3,ks,ks), bias=W8[-1], padding=ks//2), negative_slope=0.1)
    x = F.leaky_relu(F.conv2d(x, W9[:-1].view(dim3,dim3,ks,ks), bias=W9[-1], padding=ks//2), negative_slope=0.1)
    x = F.conv2d(x, W10[:-1].view(10+1,dim3,ks,ks), bias=W10[-1])
    return x

test_images, test_labels = U.read_svhn_mat('./svhn/test.mat')
len_test = len(test_images)
rp = np.random.permutation(len_test)
test_images = test_images[rp]
test_labels = test_labels[rp]

num_errors = 0
num_im_no_digit_detected = 0
num_im_for_test = 100#len(test_images)
for im_cnt, im in enumerate(test_images):
    image = torch.tensor(im/256, dtype=torch.float, device=device)
    y = model(image[None,:,:,:])[0]
    y = torch.exp(F.log_softmax(y, 0))
    
    plt.subplot(122)
    transcribed_result = []#the current transcription method is too coarse
    last_detected_label = -1
    for j in range(y.shape[2]):
        hist = np.zeros(10)
        for i in range(y.shape[1]):
            _, label = torch.max(y[:,i,j], dim=0)
            if label < 10:
                plt.text(j/y.shape[2], 1-i/y.shape[1], str(label.item()))
                hist[label.item()] += 1
                
        if max(hist)>0:# one digit is detected
            label = np.argmax(hist)
            if label!=last_detected_label:
                transcribed_result.append(label)
        else:#blank
            label = -1
            
        last_detected_label = label
        
    if transcribed_result!=test_labels[im_cnt]:
        num_errors += 1
    if transcribed_result==[]:
        num_im_no_digit_detected += 1
                
    plt.title('recognition results')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    
    plt.subplot(121)
    plt.imshow(np.transpose(im, [1,2,0]))
    plt.title('label: '+str(test_labels[im_cnt]))
    plt.axis('off')
    
    if transcribed_result!=test_labels[im_cnt]:
        plt.show()#only shows the wrong transcriptions
        time.sleep(0.1)
    else:
        plt.clf()
        
    #print(num_errors/(im_cnt+1))
    if im_cnt+1==num_im_for_test:
        break
    
print('Test sequence transcription error rate I: {}'.format(num_errors/num_im_for_test))
print('With coverage rate {}, '.format(1-num_im_no_digit_detected/num_im_for_test))
print('test sequence transcription error rate II: {}'.format((num_errors-num_im_no_digit_detected)/(num_im_for_test-num_im_no_digit_detected)))