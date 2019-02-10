# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 19:29:11 2016

@author: DIP
"""

from normalization import normalize_corpus
from utils import build_feature_matrix
import numpy as np


toy_corpus = ['The sky is blue',
'The sky is blue and beautiful',
'Look at the bright blue sky!',
'Python is a great Programming language',
'Python and Java are popular Programming languages',
'Among Programming languages, both Python and Java are the most used in Analytics',
'The fox is quicker than the lazy dog',
'The dog is smarter than the fox',
'The dog, fox and cat are good friends']

query_docs = ['The fox is definitely smarter than the dog',
            'Java is a static typed programming language unlike Python',
            'I love to relax under the beautiful blue sky!']  


# normalize and extract features from the toy corpus
norm_corpus = normalize_corpus(toy_corpus, lemmatize=True)
tfidf_vectorizer, tfidf_features = build_feature_matrix(norm_corpus,
                                                        feature_type='tfidf',
                                                        ngram_range=(1, 1), 
                                                        min_df=0.0, max_df=1.0)
                                                        
# normalize and extract features from the query corpus
norm_query_docs =  normalize_corpus(query_docs, lemmatize=True)            
query_docs_tfidf = tfidf_vectorizer.transform(norm_query_docs)

def compute_cosine_similarity(doc_features, corpus_features,
                              top_n=3):
    # get document vectors
    doc_features = doc_features.toarray()[0]
    corpus_features = corpus_features.toarray()
    # compute similarities
    similarity = np.dot(doc_features, 
                        corpus_features.T)
    # get docs with highest similarity scores
    top_docs = similarity.argsort()[::-1][:top_n]
    top_docs_with_score = [(index, round(similarity[index], 3))
                            for index in top_docs]
    return top_docs_with_score

    
print 'Document Similarity Analysis using Cosine Similarity'
print '='*60
for index, doc in enumerate(query_docs):
    
    doc_tfidf = query_docs_tfidf[index]
    top_similar_docs = compute_cosine_similarity(doc_tfidf,
                                             tfidf_features,
                                             top_n=2)
    print 'Document',index+1 ,':', doc
    print 'Top', len(top_similar_docs), 'similar docs:'
    print '-'*40 
    for doc_index, sim_score in top_similar_docs:
        print 'Doc num: {} Similarity Score: {}\nDoc: {}'.format(doc_index+1,
                                                                 sim_score,
                                                                 toy_corpus[doc_index])  
        print '-'*40       
    print                                                      
    

def compute_hellinger_bhattacharya_distance(doc_features, corpus_features,
                                            top_n=3):
    # get document vectors                                            
    doc_features = doc_features.toarray()[0]
    corpus_features = corpus_features.toarray()
    # compute hb distances
    distance = np.hstack(
                    np.sqrt(0.5 *
                            np.sum(
                                np.square(np.sqrt(doc_features) - 
                                          np.sqrt(corpus_features)), 
                                axis=1)))
    # get docs with lowest distance scores                            
    top_docs = distance.argsort()[:top_n]
    top_docs_with_score = [(index, round(distance[index], 3))
                            for index in top_docs]
    return top_docs_with_score 

print 'Document Similarity Analysis using Hellinger-Bhattacharya distance'
print '='*60
for index, doc in enumerate(query_docs):
    
    doc_tfidf = query_docs_tfidf[index]
    top_similar_docs = compute_hellinger_bhattacharya_distance(doc_tfidf,
                                             tfidf_features,
                                             top_n=2)
    print 'Document',index+1 ,':', doc
    print 'Top', len(top_similar_docs), 'similar docs:'
    print '-'*40 
    for doc_index, sim_score in top_similar_docs:
        print 'Doc num: {} Distance Score: {}\nDoc: {}'.format(doc_index+1,
                                                                 sim_score,
                                                                 toy_corpus[doc_index])  
        print '-'*40       
    print                 


import scipy.sparse as sp 

def compute_corpus_term_idfs(corpus_features, norm_corpus):
    
    dfs = np.diff(sp.csc_matrix(corpus_features, copy=True).indptr)
    dfs = 1 + dfs # to smoothen idf later
    total_docs = 1 + len(norm_corpus)
    idfs = 1.0 + np.log(float(total_docs) / dfs)
    return idfs


def compute_bm25_similarity(doc_features, corpus_features,
                            corpus_doc_lengths, avg_doc_length,
                            term_idfs, k1=1.5, b=0.75, top_n=3):
    # get corpus bag of words features
    corpus_features = corpus_features.toarray()
    # convert query document features to binary features
    # this is to keep a note of which terms exist per document
    doc_features = doc_features.toarray()[0]
    doc_features[doc_features >= 1] = 1
    
    # compute the document idf scores for present terms
    doc_idfs = doc_features * term_idfs
    # compute numerator expression in BM25 equation
    numerator_coeff = corpus_features * (k1 + 1)
    numerator = np.multiply(doc_idfs, numerator_coeff)
    # compute denominator expression in BM25 equation
    denominator_coeff =  k1 * (1 - b + 
                                (b * (corpus_doc_lengths / 
                                        avg_doc_length)))
    denominator_coeff = np.vstack(denominator_coeff)
    denominator = corpus_features + denominator_coeff
    # compute the BM25 score combining the above equations
    bm25_scores = np.sum(np.divide(numerator,
                                   denominator),
                         axis=1)
    # get top n relevant docs with highest BM25 score                     
    top_docs = bm25_scores.argsort()[::-1][:top_n]
    top_docs_with_score = [(index, round(bm25_scores[index], 3))
                            for index in top_docs]
    return top_docs_with_score

vectorizer, corpus_features = build_feature_matrix(norm_corpus,
                                                   feature_type='frequency')
query_docs_features = vectorizer.transform(norm_query_docs)

doc_lengths = [len(doc.split()) for doc in norm_corpus]   
avg_dl = np.average(doc_lengths) 
corpus_term_idfs = compute_corpus_term_idfs(corpus_features,
                                            norm_corpus)
                 
print 'Document Similarity Analysis using BM25'
print '='*60
for index, doc in enumerate(query_docs):
    
    doc_features = query_docs_features[index]
    top_similar_docs = compute_bm25_similarity(doc_features,
                                               corpus_features,
                                               doc_lengths,
                                               avg_dl,
                                               corpus_term_idfs,
                                               k1=1.5, b=0.75,
                                               top_n=2)
    print 'Document',index+1 ,':', doc
    print 'Top', len(top_similar_docs), 'similar docs:'
    print '-'*40 
    for doc_index, sim_score in top_similar_docs:
        print 'Doc num: {} BM25 Score: {}\nDoc: {}'.format(doc_index+1,
                                                                 sim_score,
                                                                 toy_corpus[doc_index])  
        print '-'*40       
    print