# CPSC-8430-Deep-Learning-001
README: HW1
AUTHOR: DINESHCHANDAR RAVICHANDRAN

The collections of files provided in this GitHub enviornment were written for
Homework 1 in Dr. Feng Luo's Deep Learning Course (CPSC-8430) 

DATE OF COMPLETION: 20TH FEB 2022

REPORT: Reports folder contains the report for this assignment in PDF. format, link: "https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/tree/main/Reports"

1.	Deep vs. Shallow:
1.1. Simulate functions. [(sin(5x).Pi(x))/5(Pi(x))] & [sgn(sin(5.Pi(x))], link: https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/blob/main/HW1/HW1_SimFunc1.ipynb
1.2. Train on actual tasks using shallow and deep models. (MNIST), link: https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/blob/main/HW1/HW1.1_MNIST.ipynb
2.	Optimization
2.1.	Visualize the optimization process. (MNIST), link: https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/blob/main/HW1/HW1.2_Visualize%20the%20Optimization%20Process_p2%20copy.ipynb
2.2. Observe gradient norm during training. [(sin(5x).Pi(x))/5(Pi(x))], link: https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/blob/main/HW1/HW1_GradientNorm.ipynb
3. Generalization (Using MNIST)
3.1. Can network fit random labels?, link: https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/blob/main/RandomM.py
3.2. Number of parameters v.s. Generalization, link: https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/blob/main/HW1/HW1.3_No_of_ParamVSGenP0.ipynb
3.3. Flatness v.s. Generalization. part-1, link: https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/blob/main/HW1/HW1.3_FlatnessVSGenP1.ipynb
3.4. Flatness v.s. Generalization. part-2, link: https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/blob/main/HW1/HW1.3_FlatnessVSGenP2.ipynb

To run these files, the following libraries are required:
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

The dataset required to run the source code is the MNIST dataset from the torchvision datasets library in pytorch. Once the dataset is downloaded to this 
pathfile, it can be used by every other file and is the only file that needs to be downloaded. This has been codded in the above assignmets, in every file that requires the MNIST dataset, there is a
line of code that grabs the dataset. It is provided below: train_dataset = datasets.MNIST('', train=True, download=True/False, ...)
To download the dataset, change the download option to True. This is done in the source code of HW1.1_MNIST.ipynb.
You will need internet connect to download the dataset.

All the assignments task are performed on juyper notbooks and saved in .ipynb files,which has all the required packages setup in the enviornment and run the cells in the file from top to bottom. 

For Random Mnist the assignment was perfomed on PALMETTO CLUSTER with mem=5gb,ncpus=16, link: https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/blob/main/RandomM.py

Incase of any queries or issues please contact me,

E-mail: dravich@g.clemson.edu

Thanks,
Dineshchandar Ravichandran
