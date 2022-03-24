#!/usr/bin/env python
# coding: utf-8

# In[5]:


"""
    This is script contains Test and BLEU eval for CPSC8830-Deep Learning HW02  S2VT assign.
    @author: Dineshchanadar Ravichandran
    Email: dravich@clemson.edu 

"""

# import libraries
import sys
import torch
import json
from torch.utils.data import DataLoader
from bleu_eval import BLEU
import Main
import numpy as np
import pickle


# In[9]:


if not torch.cuda.is_available():
    modelIP = torch.load('SavedModel/model1.h5', map_location=lambda storage, loc: storage)
else:
    modelIP = torch.load('SavedModel/model1.h5')


# In[10]:


files_dir = 'testing_data/feat'
i2w,w2i,dictonary = Main.dictonaryFunc()

test_dataset = Main.test_dataloader('{}'.format(sys.argv[1]))
# test_dataset = Main.test_dataloader(files_dir)

test_dataloader = Main.DataLoader(dataset = test_dataset, batch_size=1, shuffle=True, num_workers=8)

# with open('i2wData.pickle', 'rb') as f:
#     i2w = pickle.load(f)

model = modelIP.cuda()

ss = Main.test(test_dataloader, model, i2w)

with open(sys.argv[2], 'w') as f:
    for id, s in ss:
        f.write('{},{}\n'.format(id, s))


# In[11]:


# Bleu Eval
test = json.load(open('testing_label.json','r'))
#output = 'testing_data.txt'
output = sys.argv[2]
result = {}

with open(output,'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma+1:]
        result[test_id] = caption
#count by the method described in the paper https://aclanthology.info/pdf/P/P02/P02-1040.pdf
bleu=[]
for item in test:
    score_per_video = []
    captions = [x.rstrip('.') for x in item['caption']]
    score_per_video.append(BLEU(result[item['id']],captions,True))
    bleu.append(score_per_video[0])
average = sum(bleu) / len(bleu)
print("Average bleu score is ", np.round(average,3))


# In[ ]:





# In[ ]:




