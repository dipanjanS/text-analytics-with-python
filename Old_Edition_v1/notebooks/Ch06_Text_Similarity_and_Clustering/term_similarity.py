# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 04:02:10 2016

@author: DIP
"""

import numpy as np
from scipy.stats import itemfreq

def vectorize_terms(terms):
    terms = [term.lower() for term in terms]
    terms = [np.array(list(term)) for term in terms]
    terms = [np.array([ord(char) for char in term]) 
                for term in terms]
    return terms
    
def boc_term_vectors(word_list):
    word_list = [word.lower() for word in word_list]
    unique_chars = np.unique(
                        np.hstack([list(word) 
                        for word in word_list]))
    word_list_term_counts = [{char: count for char, count in itemfreq(list(word))}
                             for word in word_list]
    
    boc_vectors = [np.array([int(word_term_counts.get(char, 0)) 
                            for char in unique_chars])
                   for word_term_counts in word_list_term_counts]
    return list(unique_chars), boc_vectors
                             
                             
root = 'Believe'
term1 = 'beleive'
term2 = 'bargain'
term3 = 'Elephant'    

terms = [root, term1, term2, term3]

vec_root, vec_term1, vec_term2, vec_term3 = vectorize_terms(terms)
print '''
root: {}
term1: {}
term2: {}
term3: {}
'''.format(vec_root, vec_term1, vec_term2, vec_term3)

features, (boc_root, boc_term1, boc_term2, boc_term3) = boc_term_vectors(terms)
print 'Features:', features
print '''
root: {}
term1: {}
term2: {}
term3: {}
'''.format(boc_root, boc_term1, boc_term2, boc_term3)



def hamming_distance(u, v, norm=False):
    if u.shape != v.shape:
        raise ValueError('The vectors must have equal lengths.')
    return (u != v).sum() if not norm else (u != v).mean()
    
def manhattan_distance(u, v, norm=False):
    if u.shape != v.shape:
        raise ValueError('The vectors must have equal lengths.')
    return abs(u - v).sum() if not norm else abs(u - v).mean()

def euclidean_distance(u,v):
    if u.shape != v.shape:
        raise ValueError('The vectors must have equal lengths.')
    distance = np.sqrt(np.sum(np.square(u - v)))
    return distance

import copy
import pandas as pd

def levenshtein_edit_distance(u, v):
    # convert to lower case
    u = u.lower()
    v = v.lower()
    # base cases
    if u == v: return 0
    elif len(u) == 0: return len(v)
    elif len(v) == 0: return len(u)
    # initialize edit distance matrix
    edit_matrix = []
    # initialize two distance matrices 
    du = [0] * (len(v) + 1)
    dv = [0] * (len(v) + 1)
    # du: the previous row of edit distances
    for i in range(len(du)):
        du[i] = i
    # dv : the current row of edit distances    
    for i in range(len(u)):
        dv[0] = i + 1
        # compute cost as per algorithm
        for j in range(len(v)):
            cost = 0 if u[i] == v[j] else 1
            dv[j + 1] = min(dv[j] + 1, du[j + 1] + 1, du[j] + cost)
        # assign dv to du for next iteration
        for j in range(len(du)):
            du[j] = dv[j]
        # copy dv to the edit matrix
        edit_matrix.append(copy.copy(dv))
    # compute the final edit distance and edit matrix    
    distance = dv[len(v)]
    edit_matrix = np.array(edit_matrix)
    edit_matrix = edit_matrix.T
    edit_matrix = edit_matrix[1:,]
    edit_matrix = pd.DataFrame(data=edit_matrix,
                               index=list(v),
                               columns=list(u))
    return distance, edit_matrix
    
def cosine_distance(u, v):
    distance = 1.0 - (np.dot(u, v) / 
                        (np.sqrt(sum(np.square(u))) * np.sqrt(sum(np.square(v))))
                     )
    return distance


# DEMOS!

# build the term vectors here    
root_term = root
root_vector = vec_root
root_boc_vector = boc_root

terms = [term1, term2, term3]
vector_terms = [vec_term1, vec_term2, vec_term3]
boc_vector_terms = [boc_term1, boc_term2, boc_term3]


# HAMMING DISTANCE DEMO
for term, vector_term in zip(terms, vector_terms):
    print 'Hamming distance between root: {} and term: {} is {}'.format(root_term,
                                                                term,
                                                                hamming_distance(root_vector, vector_term, norm=False))


for term, vector_term in zip(terms, vector_terms):
    print 'Normalized Hamming distance between root: {} and term: {} is {}'.format(root_term,
                                                                term,
                                                                round(hamming_distance(root_vector, vector_term, norm=True), 2))


# MANHATTAN DISTANCE DEMO
for term, vector_term in zip(terms, vector_terms):
    print 'Manhattan distance between root: {} and term: {} is {}'.format(root_term,
                                                                term,
                                                                manhattan_distance(root_vector, vector_term, norm=False))

for term, vector_term in zip(terms, vector_terms):
    print 'Normalized Manhattan distance between root: {} and term: {} is {}'.format(root_term,
                                                                term,
                                                                round(manhattan_distance(root_vector, vector_term, norm=True),2))


# EUCLIDEAN DISTANCE DEMO
for term, vector_term in zip(terms, vector_terms):
    print 'Euclidean distance between root: {} and term: {} is {}'.format(root_term,
                                                                term,
                                                                round(euclidean_distance(root_vector, vector_term),2))


# LEVENSHTEIN EDIT DISTANCE DEMO
for term in terms:
    edit_d, edit_m = levenshtein_edit_distance(root_term, term)
    print 'Computing distance between root: {} and term: {}'.format(root_term,
                                                                    term)
    print 'Levenshtein edit distance is {}'.format(edit_d)
    print 'The complete edit distance matrix is depicted below'
    print edit_m
    print '-'*30                                                                             


# COSINE DISTANCE\SIMILARITY DEMO
for term, boc_term in zip(terms, boc_vector_terms):
    print 'Analyzing similarity between root: {} and term: {}'.format(root_term,
                                                                      term)
    distance = round(cosine_distance(root_boc_vector, boc_term),2)
    similarity = 1 - distance                                                           
    print 'Cosine distance  is {}'.format(distance)
    print 'Cosine similarity  is {}'.format(similarity)
    print '-'*40
                                                                

