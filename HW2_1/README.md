CPSC 8430 Deep Learning Homework 2 : Video Caption Generation using S2VT (H1)

To execute the S2VT, kindly follow the following steps: (H2)
First item Clone this repository
Second item Add the test features adhereing to folder structure "testing_data/feat" and add the testing labels in JSON format at "testing_data/" with the name "testing_label.json".
Third item Edit the shell script "hw2_seq2seq.sh" and replace the "$1" with "testing_data" and "$2" with the oput .txt file location.
Forth item The testing is performed using pretrianed model stored in "LSTM/SavedModel/model0.h5". Which is pretained with the training data proved by the professor, with the following parameters:

first sub item Batch Size = 10 (Trained with 126, 32, 10 where Batch size = 10 was generating lowest loss).
second sub item Epochs = 100
third sub item Learning rate =0.0001
forth sub item Loss Function = nn.CrossEntropyLoss()
fifth sub item Optimizer = Adam.
sixth sub item Training Sample Size = 1450
seventh sub item Video features dimension=4096
eighth sub item Video frame dimension=80

Result: (H2)
Best bleu score is 0.709, for the testdata present in "/testing_data"

System Requirment: (H2)
The code was trained on Palmetto system with the following configuration:
CPU: 20
Mem: 125
GPU: 1
GPU model: Any
Execution Time: 12371.845 seconds

Report: 

Incase of any issues/quereis kindly contact me at dravich@g.clemson.edu

Thanks,

Dineschandar Ravichandran.
