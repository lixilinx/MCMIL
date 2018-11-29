import random
import torch
import utilities as U
import torch.nn.functional as F

C = 10#number of classes
Dim = random.randint(1,3)
if Dim==1:
    x = torch.randn(C+1, 19)#1D, e.g., audio
elif Dim==2:
    x = torch.randn(C+1, 13,17)#2D, e.g., image
else:
    x = torch.randn(C+1, 13,19,17)#3D, e.g., video
x[-1] = 5
logP = F.log_softmax(x, dim=0).double()#propose to use double precision for log(P)


"""Compare general and manually expanded equations for log Prb({labels}|P)
They should yied the same results"""
print(U.log_prb_labels(logP, []), U.log_prb_0label(logP))
print(U.log_prb_labels(logP, [8]), U.log_prb_1label(logP, 8))
print(U.log_prb_labels(logP, [1,6]), U.log_prb_2label(logP, [6,1]))
print(U.log_prb_labels(logP, [4,0,2]), U.log_prb_3label(logP, [2,4,0]))
print(U.log_prb_labels(logP, [1,5,0,2]), U.log_prb_4label(logP, [5,2,1,0]))
print(U.log_prb_labels(logP, [1,5,0,9,2]), U.log_prb_5label(logP, [5,9,2,1,0]))
print(U.log_prb_labels(logP, [1,3,5,0,9,2]), U.log_prb_6label(logP, [5,9,2,1,3,0]))

"""The one-label-per-sample code assume one more dimension for batch processing
Just insert None, and that dimension is 1"""
print(U.log_prb_labels(logP, [7]), U.log_prb_1l_per_smpl(logP[None], [7]))

"""Code using manually expanded equations is faster than the code using itertools lib""" 
import time
t0=time.time()
for i in range(100):
    U.log_prb_6label(logP, [5,9,2,1,3,0])
print('wall time of the code with manually expanded eq.: {}'.format(time.time()-t0))

t0=time.time()
for i in range(100):
    U.log_prb_labels(logP, [5,9,2,1,3,0])
print('wall time of the code with itertools lib: {}'.format(time.time()-t0))