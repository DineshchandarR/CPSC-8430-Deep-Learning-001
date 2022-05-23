# CPSC-8430-Deep-Learning-001
AUTHOR: DINESHCHANDAR RAVICHANDRAN

The collections of files provided in this GitHub enviornment were written for
Homework 1-3 in Dr. Feng Luo's Deep Learning Course (CPSC-8430) 

REPORT: Reports folder contains the report for this assignment in PDF format.

## Homework-1:
1.	Deep vs. Shallow:
1.1. Simulate functions. [(sin(5x).Pi(x))/5(Pi(x))] & [sgn(sin(5.Pi(x))], link: https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/blob/main/HW1/HW1_SimFunc1.ipynb
> Concepts covered:
     # Creating function in tesor.
     # Creation and functioning of DNN.
     # Activation function.
     # Loss function and Optimization function.
     # Foward and Backward propogation.
     # Calculation of Loss and Accuracy using predcition.
     # Visvulazing model performance.
   
1.2. Train on actual tasks using shallow and deep models. (MNIST), link: https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/blob/main/HW1/HW1.1_MNIST.ipynb
> Concepts covered:
    # Convolution Neural Network.
    # Pooling (Max-Pooling).
    # Fully connected layer.
    # Dropout and weight decay.
    # Understanding the relation between network complexity and performance.

2.	Optimization
2.1.	Visualize the optimization process. (MNIST), link: https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/blob/main/HW1/HW1.2_Visualize%20the%20Optimization%20Process_p2%20copy.ipynb
> Concepts covered:
    # Extraction of weights at epoch, layer level.
    # Weights to vector conversion.
    # Performing PCA for dimensional reduction.
    # Visualization of optimization. 
    
2.2. Observe gradient norm during training. [(sin(5x).Pi(x))/5(Pi(x))], link: https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/blob/main/HW1/HW1_GradientNorm.ipynb
> Concepts covered:
    # Observing gradient norm
    # Weights to vector conversion.
    # Performing PCA for dimensional reduction.
    # Visualization of optimization.
    
2.3. What Happened When Gradient is Almost Zero!, link: https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/blob/main/HW1/HW1.2_GradZero_v2.ipynb
Finding grad zero ratio using hessian matrix.
    
3. Generalization (Using MNIST)
3.1. Can network fit random labels?, link: https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/blob/main/RandomM.py
> Concepts covered:
    # To oberserve the networks performance when trained on randomized labels and its performance against test data with proper label values
    # We can see in this case the network is not learning rather memorizing to reduce loss. Hence performance against test gets worse as the loss reduces in training.
    
3.2. Number of parameters v.s. Generalization, link: https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/blob/main/HW1/HW1.3_No_of_ParamVSGenP0.ipynb
> Concpets covered:
    # To understand the effects of parameter size of a network on its performance.
    
3.3. Flatness v.s. Generalization. part-1, link: https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/blob/main/HW1/HW1.3_FlatnessVSGenP1.ipynb
> Concepts covered:
    # To visualiaize models capabilty for generalization on applying the interpolated the weights of 2 Deep Learning models to a new model.

3.4. Flatness v.s. Generalization. part-2, link: https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/blob/main/HW1/HW1.3_FlatnessVSGenP2.ipynb
>Concepts covered:
    # To understand the relation between sensitivity and batch size, based on the training of 2 models on MNIST dataset, and to observe the decrease in sensitivity as batch size increases.
    
The dataset required to run the source code is the MNIST dataset from the torchvision datasets library in pytorch. Once the dataset is downloaded to this 
pathfile, it can be used by every other file and is the only file that needs to be downloaded. This has been codded in the above assignmets, in every file that requires the MNIST dataset, there is a
line of code that grabs the dataset. It is provided below: train_dataset = datasets.MNIST('', train=True, download=True/False, ...)
To download the dataset, change the download option to True. This is done in the source code of HW1.1_MNIST.ipynb.
You will need internet connect to download the dataset.

All the assignments task are performed on juyper notbooks and saved in .ipynb files,which has all the required packages setup in the enviornment and run the cells in the file from top to bottom. 


## Homework 2| Video Caption Generation using S2VT :
Created a Sequence to Sequence - Video to Text using GRU, train the model using the provided video files and the caption files. Test the model and evaluate performance using BLEU evaluation. Current model BLEU score= 0.709.

## Homework 3|  Implementation of Generative Adversarial Networks :
To train a discriminator/generator pair on the CIFAR10 dataset utilizing techniques from DCGAN, Wasserstein GANs, and ACGAN. Post which to evaluate and compare the performance between the DCGAN, WGAN, WGAN-GP  & ACGAN for 50 Epoch using FID score, as illustrated below:

![image](https://user-images.githubusercontent.com/96357078/164615269-6e169c7d-7858-4359-b409-d28cf1c216c4.png)

### Package Required
1. Python:3.9.7
2. torch
3. torch.nn 
4. torchvision
5. torchvision.transforms
7. matplotlib.pyplot
8. numpy
9. torch.nn.utils import parameters_to_vector, vector_to_parameters
10. torch.autograd import Variable
11. pandas
12. copy
13. sklearn.decomposition import PCA
14. torch.optim as optim
15. torch.nn.functional
16. torchvision.datasets as datasets
17. torchvision.transforms as transforms
18. torchvision.utils as utils
19. matplotlib.animation as animation
20. Python.display import HTML
21. time
22. torch.utils.data import Subset
23. torchvision.models as models
24. torch.nn.functional as F
25. from scipy import linalg

## Reports
All Homework final reports can be found at :
https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/tree/main/Reports

## Resource
#### HW1
For Random Mnist the assignment was perfomed on PALMETTO CLUSTER with mem=5gb,ncpus=16.

#### HW2
The code was trained on Palmetto system with the following configuration: *CPU: 20 *Mem: 125 *GPU: 1 *GPU model: Any *Execution Time: 12371.845 seconds

---------------
Incase of any queries or issues please contact me,

E-mail: dravich@g.clemson.edu

Thanks,
Dineshchandar Ravichandran
