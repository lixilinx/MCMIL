### Multiclass multiple instance learning

Please check function [log_prb_labels()](https://github.com/lixilinx/MCMIL/blob/04ad08465d45b2ee8d6174eb1f76dc809550eb16/utilities.py#L6) for logarithm bag probability calculation. Although we just demonstrate its application to multiple object detection and localization, this function accepts probability tensor P with other orders, as long as the first dimension is number_of_classes + 1.    

We have used [this optimization code](https://github.com/lixilinx/psgd_torch/blob/master/preconditioned_stochastic_gradient_descent.py) to train our models. You may try your favorite optimization methods.

We demonstrated its applications to image recognition and street view house number transcription using all convolutional networks. For MNIST, we got test classification error rate slightly higher than 0.3%. For SVHN, we got sequence transcription error rate slightly higher than 5% with a model having about 7.5M coefficients (vs. about 4% by [this specialized model](https://arxiv.org/abs/1312.6082) having about 51M coefficients and trained with detailed annotations). Our learned model might be able to directly read the numbers in many of the original images as shown below. Preprocessed SVHN dataset and some pretrained models are [here](https://drive.google.com/drive/folders/1BvVpUw3OY3RtkJo-bfujwBsa0y6e49lZ).

![alt text](https://github.com/lixilinx/Learning_with_Labels_of_Existing-Nonexisting/blob/master/misc/svhn_test.png)
