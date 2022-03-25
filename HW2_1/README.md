CPSC 8430 Deep Learning Homework 2 : Video Caption Generation using S2VT 
============= 

To execute the S2VT, kindly follow the following steps: 
---------------
First item Clone this repository
Second item Add the test features adhereing to folder structure "testing_data/feat" and add the testing labels in JSON format at "testing_data/" with the name "testing_label.json".
Third item Edit the shell script "hw2_seq2seq.sh" and replace the "$1" with "testing_data" and "$2" with the oput .txt file location.
Forth item The testing is performed using pretrianed model stored in "LSTM/SavedModel/model0.h5". Which is pretained with the training data proved by the professor, with the following parameters:

* Batch Size = 10 (Trained with 126, 32, 10 where Batch size = 10 was generating lowest loss).
* Epochs = 100
* Learning rate =0.0001
  * Loss Function = nn.CrossEntropyLoss()
  * Optimizer = Adam.
  * Training Sample Size = 1450
  * Video features dimension=4096
  * Video frame dimension=80

Result:
---------------
Best bleu score is 0.709, for the testdata present in "/testing_data"

System Requirment:
---------------
The code was trained on Palmetto system with the following configuration:
*CPU: 20
*Mem: 125
*GPU: 1
*GPU model: Any
*Execution Time: 12371.845 seconds

Report: 
---------------
https://github.com/DineshchandarR/CPSC-8430-Deep-Learning-001/blob/main/HW2_1/Ravichandran_Dineshchandar_HW2.pdf


Incase of any issues/quereis kindly contact me at dravich@g.clemson.edu

Thanks,

Dineschandar Ravichandran.
