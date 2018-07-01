# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 01:27:49 2016

@author: DIP
"""

# download dataset from http://ai.stanford.edu/~amaas/data/sentiment/

# change the directory path to your path 
# path should point to the unzipped directory with reviews
# I had unzipped it to E:/aclImdb

import os
import pandas as pd
import numpy as np

labels = {'pos': 'positive', 'neg': 'negative'}

dataset = pd.DataFrame()
for directory in ('test', 'train'):
    for sentiment in ('pos', 'neg'):
        path =r'E:/aclImdb/{}/{}'.format(directory, sentiment)
        for review_file in os.listdir(path):
            with open(os.path.join(path, review_file), 'r') as input_file:
                review = input_file.read()
            dataset = dataset.append([[review, labels[sentiment]]], 
                                     ignore_index=True)

dataset.columns = ['review', 'sentiment']

indices = dataset.index.tolist()
np.random.shuffle(indices)
indices = np.array(indices)

dataset = dataset.reindex(index=indices)

dataset.to_csv('movie_reviews.csv', index=False)