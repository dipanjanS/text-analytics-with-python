# -*- coding: utf-8 -*-
"""
Created on Sat Sep 03 09:06:06 2016

@author: DIP
"""

from gensim import corpora, models
from normalization import normalize_corpus
import numpy as np

toy_corpus = ["The fox jumps over the dog",
"The fox is very clever and quick",
"The dog is slow and lazy",
"The cat is smarter than the fox and the dog",
"Python is an excellent programming language",
"Java and Ruby are other programming languages",
"Python and Java are very popular programming languages",
"Python programs are smaller than Java programs"]

# LSI topic model
norm_tokenized_corpus = normalize_corpus(toy_corpus, tokenize=True)
norm_tokenized_corpus

dictionary = corpora.Dictionary(norm_tokenized_corpus)
print dictionary.token2id

corpus = [dictionary.doc2bow(text) for text in norm_tokenized_corpus]
corpus

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

total_topics = 2

lsi = models.LsiModel(corpus_tfidf, 
                      id2word=dictionary, 
                      num_topics=total_topics)
                      
for index, topic in lsi.print_topics(total_topics):
    print 'Topic #'+str(index+1)
    print topic
    print              
    

def print_topics_gensim(topic_model, total_topics=1,
                        weight_threshold=0.0001,
                        display_weights=False,
                        num_terms=None):
    
    for index in range(total_topics):
        topic = topic_model.show_topic(index)
        topic = [(word, round(wt,2)) 
                 for word, wt in topic 
                 if abs(wt) >= weight_threshold]
        if display_weights:
            print 'Topic #'+str(index+1)+' with weights'
            print topic[:num_terms] if num_terms else topic
        else:
            print 'Topic #'+str(index+1)+' without weights'
            tw = [term for term, wt in topic]
            print tw[:num_terms] if num_terms else tw
        print
    

print_topics_gensim(topic_model=lsi,
                    total_topics=total_topics,
                    num_terms=5,
                    display_weights=True)

    
# LSI custom built topic model    
from utils import build_feature_matrix, low_rank_svd

norm_corpus = normalize_corpus(toy_corpus)

vectorizer, tfidf_matrix = build_feature_matrix(norm_corpus, 
                                    feature_type='tfidf')
td_matrix = tfidf_matrix.transpose()
                              
td_matrix = td_matrix.multiply(td_matrix > 0)

total_topics = 2
feature_names = vectorizer.get_feature_names()

u, s, vt = low_rank_svd(td_matrix, singular_count=total_topics)
weights = u.transpose() * s[:, None]

def get_topics_terms_weights(weights, feature_names):
    feature_names = np.array(feature_names)
    sorted_indices = np.array([list(row[::-1]) 
                           for row 
                           in np.argsort(np.abs(weights))])
    sorted_weights = np.array([list(wt[index]) 
                               for wt, index 
                               in zip(weights,sorted_indices)])
    sorted_terms = np.array([list(feature_names[row]) 
                             for row 
                             in sorted_indices])
    
    topics = [np.vstack((terms.T, 
                     term_weights.T)).T 
              for terms, term_weights 
              in zip(sorted_terms, sorted_weights)]     
    
    return topics            
  
                       
def print_topics_udf(topics, total_topics=1,
                     weight_threshold=0.0001,
                     display_weights=False,
                     num_terms=None):
    
    for index in range(total_topics):
        topic = topics[index]
        topic = [(term, float(wt))
                 for term, wt in topic]
        topic = [(word, round(wt,2)) 
                 for word, wt in topic 
                 if abs(wt) >= weight_threshold]
                     
        if display_weights:
            print 'Topic #'+str(index+1)+' with weights'
            print topic[:num_terms] if num_terms else topic
        else:
            print 'Topic #'+str(index+1)+' without weights'
            tw = [term for term, wt in topic]
            print tw[:num_terms] if num_terms else tw
        print
topics = get_topics_terms_weights(weights, feature_names)        
print_topics_udf(topics=topics,
                 total_topics=total_topics,
                 weight_threshold=0.15,
                 display_weights=False)

def train_lsi_model_gensim(corpus, total_topics=2):
    
    norm_tokenized_corpus = normalize_corpus(corpus, tokenize=True)
    dictionary = corpora.Dictionary(norm_tokenized_corpus)
    mapped_corpus = [dictionary.doc2bow(text) 
                     for text in norm_tokenized_corpus]
    tfidf = models.TfidfModel(mapped_corpus)
    corpus_tfidf = tfidf[mapped_corpus]
    lsi = models.LsiModel(corpus_tfidf, 
                          id2word=dictionary,
                          num_topics=total_topics)
    return lsi
 




def train_lda_model_gensim(corpus, total_topics=2):
    
    norm_tokenized_corpus = normalize_corpus(corpus, tokenize=True)
    dictionary = corpora.Dictionary(norm_tokenized_corpus)
    mapped_corpus = [dictionary.doc2bow(text) 
                     for text in norm_tokenized_corpus]
    tfidf = models.TfidfModel(mapped_corpus)
    corpus_tfidf = tfidf[mapped_corpus]
    lda = models.LdaModel(corpus_tfidf, 
                          id2word=dictionary,
                          iterations=1000,
                          num_topics=total_topics)
    return lda                     



lda_gensim = train_lda_model_gensim(toy_corpus,
                                    total_topics=2)
                                    
print_topics_gensim(topic_model=lda_gensim,
                    total_topics=2,
                    num_terms=5,
                    display_weights=True)                                    

                     
from sklearn.decomposition import LatentDirichletAllocation

norm_corpus = normalize_corpus(toy_corpus)
vectorizer, tfidf_matrix = build_feature_matrix(norm_corpus, 
                                    feature_type='tfidf')                     
total_topics = 2
lda = LatentDirichletAllocation(n_topics=total_topics, 
                                max_iter=1000,
                                learning_method='online', 
                                learning_offset=50.,
                                random_state=42)
lda.fit(tfidf_matrix)

feature_names = vectorizer.get_feature_names()
weights = lda.components_

topics = get_topics_terms_weights(weights, feature_names)
print_topics_udf(topics=topics,
                 total_topics=total_topics,
                 num_terms=8,
                 display_weights=True)
                 
                 
from sklearn.decomposition import NMF

norm_corpus = normalize_corpus(toy_corpus)
vectorizer, tfidf_matrix = build_feature_matrix(norm_corpus, 
                                    feature_type='tfidf')                     
total_topics = 2
nmf = NMF(n_components=total_topics, 
          random_state=42, alpha=.1, l1_ratio=.5)
nmf.fit(tfidf_matrix)      

feature_names = vectorizer.get_feature_names()
weights = nmf.components_

topics = get_topics_terms_weights(weights, feature_names)
print_topics_udf(topics=topics,
                 total_topics=total_topics,
                 num_terms=None,
                 display_weights=True)   
                 
import pandas as pd
import numpy as np 
                 
CORPUS = pd.read_csv('amazon_skyrim_reviews.csv')                     
CORPUS = np.array(CORPUS['Reviews'])

# view sample review
print CORPUS[12]

        
total_topics = 5
        
lsi_gensim = train_lsi_model_gensim(CORPUS,
                                    total_topics=total_topics)
print_topics_gensim(topic_model=lsi_gensim,
                    total_topics=total_topics,
                    num_terms=10,
                    display_weights=False) 


lda_gensim = train_lda_model_gensim(CORPUS,
                                    total_topics=total_topics)
print_topics_gensim(topic_model=lda_gensim,
                    total_topics=total_topics,
                    num_terms=10,
                    display_weights=False) 


norm_corpus = normalize_corpus(CORPUS)
vectorizer, tfidf_matrix = build_feature_matrix(norm_corpus, 
                                    feature_type='tfidf') 
feature_names = vectorizer.get_feature_names()


lda = LatentDirichletAllocation(n_topics=total_topics, 
                                max_iter=1000,
                                learning_method='online', 
                                learning_offset=10.,
                                random_state=42)
lda.fit(tfidf_matrix)
weights = lda.components_
topics = get_topics_terms_weights(weights, feature_names)
print_topics_udf(topics=topics,
                 total_topics=total_topics,
                 num_terms=10,
                 display_weights=False)


nmf = NMF(n_components=total_topics, 
          random_state=42, alpha=.1, l1_ratio=.5)
nmf.fit(tfidf_matrix)      

feature_names = vectorizer.get_feature_names()
weights = nmf.components_

topics = get_topics_terms_weights(weights, feature_names)
print_topics_udf(topics=topics,
                 total_topics=total_topics,
                 num_terms=10,
                 display_weights=False)                                       