#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import torch

from collections import defaultdict

from src.dataset import Bigrammer
from src.dataset_reader import sentence2words_preprocessing


# In[2]:


bigrammer, w2v = torch.load("models/embedding.pth")
w2v.to('cpu')


# In[3]:


v_size = len(bigrammer.word2idx)
m,_ = torch.max(torch.abs(w2v.weight), dim=0)


# In[5]:


def get_norm_vec(word):
    id_tensor = torch.LongTensor([bigrammer.word2idx[word]])
    word_vec = w2v(id_tensor)
    return word_vec / m

# In[6]:


test_set = pd.read_csv("data/google-analogies.csv", index_col=0)


# In[7]:


columns = test_set.columns
vals = test_set.values


# In[8]:


pr, not_pr = 0, 0
clean_set = defaultdict(list)
for val in vals:
    cat, words = val[0], val[1:]
    words = [w.lower() for w in words]
    # check if all are present
    if all([w in bigrammer.word2idx for w in words]):
        pr += 1
        clean_set[cat].append(words)
    else:
        not_pr += 1

print("Test cases: {} present / {} not present".format(pr, not_pr))


# # 3CosAdd by categories

# In[9]:


def cos(a,b):
    a = a.flatten()
    b = b.flatten()
    return a @ b / (a.norm() * b.norm())


a = torch.tensor([1.0, 0.0]).view(1,2).float()
b = torch.tensor([1.0, 0.5]).view(1,2).float()

cos(a,b)


# In[10]:


distances = {} # category -> list of cosine dists
for cat, samples in clean_set.items():
    distances[cat] = []
    for case in samples:
        # 1. get all 4 vectors:
        vecs = [get_norm_vec(w) for w in case]
        # 2. calculate distance to target (case[3])
        target = vecs[3]
        destination = vecs[2] + (vecs[1] - vecs[0])
        distances[cat].append(abs(cos(target, destination).item()))


# In[32]:


all_dists = []
cats_mean_dists = []
def print_stat(metrics):
    print("\t max:\t{:.3f}".format(metrics[0]))
    print("\t mean:\t{:.3f}".format(metrics[1]))
    print("\t std:\t{:.3f}".format(metrics[2]))
    
for cat, cs in distances.items():
    print(cat)
    metrics = (np.max(cs), np.mean(cs), np.std(cs))
    print_stat(metrics)
    all_dists += cs
    cats_mean_dists.append(metrics)


# In[35]:


print("Mean metrics between categories:")
metrics = np.array(cats_mean_dists).mean(axis=0)
print_stat(metrics)


# In[36]:


print("For all test set:")
cs = all_dists
print_stat((np.max(cs), np.mean(cs), np.std(cs)))

