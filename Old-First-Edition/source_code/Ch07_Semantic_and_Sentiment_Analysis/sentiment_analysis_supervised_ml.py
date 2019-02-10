# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 02:04:30 2016

@author: DIP
"""

import pandas as pd
import numpy as np
from normalization import normalize_corpus
from utils import build_feature_matrix


dataset = pd.read_csv(r'E:/aclImdb/movie_reviews.csv')

print dataset.head()

train_data = dataset[:35000]
test_data = dataset[35000:]

train_reviews = np.array(train_data['review'])
train_sentiments = np.array(train_data['sentiment'])
test_reviews = np.array(test_data['review'])
test_sentiments = np.array(test_data['sentiment'])


sample_docs = [100, 5817, 7626, 7356, 1008, 7155, 3533, 13010]
sample_data = [(test_reviews[index],
                test_sentiments[index])
                  for index in sample_docs]

sample_data    

# normalization
norm_train_reviews = normalize_corpus(train_reviews,
                                      lemmatize=True,
                                      only_text_chars=True)
# feature extraction                                                                            
vectorizer, train_features = build_feature_matrix(documents=norm_train_reviews,
                                                  feature_type='tfidf',
                                                  ngram_range=(1, 1), 
                                                  min_df=0.0, max_df=1.0)                                      
                                      
                                      

from sklearn.linear_model import SGDClassifier
# build the model
svm = SGDClassifier(loss='hinge', n_iter=500)
svm.fit(train_features, train_sentiments)



# normalize reviews                        
norm_test_reviews = normalize_corpus(test_reviews,
                                     lemmatize=True,
                                     only_text_chars=True)  
# extract features                                     
test_features = vectorizer.transform(norm_test_reviews)         


for doc_index in sample_docs:
    print 'Review:-'
    print test_reviews[doc_index]
    print 'Actual Labeled Sentiment:', test_sentiments[doc_index]
    doc_features = test_features[doc_index]
    predicted_sentiment = svm.predict(doc_features)[0]
    print 'Predicted Sentiment:', predicted_sentiment
    print
   

predicted_sentiments = svm.predict(test_features)       

from utils import display_evaluation_metrics, display_confusion_matrix, display_classification_report

display_evaluation_metrics(true_labels=test_sentiments,
                           predicted_labels=predicted_sentiments,
                           positive_class='positive')  
                           
display_confusion_matrix(true_labels=test_sentiments,
                         predicted_labels=predicted_sentiments,
                         classes=['positive', 'negative'])
                         
display_classification_report(true_labels=test_sentiments,
                              predicted_labels=predicted_sentiments,
                              classes=['positive', 'negative'])                         