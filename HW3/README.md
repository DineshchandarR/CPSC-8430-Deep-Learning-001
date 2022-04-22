CPSC 8430 Deep Learning Homework 3 : Implementaion of Generative Adversarial Network’s
============= 

Probelem Statement 
---------------
Train a discriminator/generator pair on the CIFAR10 dataset utilizing techniques from DCGAN, Wasserstein GANs, and ACGAN

Package Required
---------------
torch
torchvision
numpy as np
torch.nn as nn
torch.optim as optim
torch.nn.functional as F
torchvision.datasets as datasets
torchvision.transforms as transforms
torchvision.utils as utils
matplotlib.pyplot as plt
matplotlib.animation as animation
Python.display import HTML
time
torch.utils.data import Subset
torchvision.models as models
torch.nn.functional as F
from scipy import linalg
pandas as pd



Result:
---------------
Best result was obtained from DCGAN following is the FIDscore graph for DCGAN, WGAN, WGAN-GP  & ACGAN for 50 Epoch

![image](https://user-images.githubusercontent.com/96357078/164615269-6e169c7d-7858-4359-b409-d28cf1c216c4.png)

Report: 
---------------
https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/blob/main/HW3/Ravichandran_Dineshchandar_HW3.pdf

---------------
Incase of any issues/quereis kindly contact me at dravich@g.clemson.edu

Thanks,

Dineschandar Ravichandran.
