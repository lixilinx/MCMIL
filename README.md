# Learning with Labels of Existing/Nonexisting
A weak supervision/label learning method. Only need to know whether any instance of a class exists or not in a sample.

We used optimization code https://github.com/lixilinx/psgd_torch/blob/master/preconditioned_stochastic_gradient_descent.py to train our models. You may try your favorite optimization methods.

We demonstrated its application to street view house number transcription using all convolutional networks. You may need to download and preprocess the SVHN dataset. The learned CNN model can directly read numbers in the images, as shown below.

![alt text](https://github.com/lixilinx/Learning_with_Labels_of_Existing-Nonexisting/blob/master/misc/svhn_test.png)
