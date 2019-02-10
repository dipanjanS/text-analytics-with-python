# -*- coding: utf-8 -*-
"""
Created on Sat Sep 03 19:33:32 2016

@author: DIP
"""

from nltk.corpus import gutenberg
from normalization import normalize_corpus
import nltk
from operator import itemgetter

alice = gutenberg.sents(fileids='carroll-alice.txt')
alice = [' '.join(ts) for ts in alice]
norm_alice = filter(None, normalize_corpus(alice, lemmatize=False))

# print first line
print norm_alice[0]

def flatten_corpus(corpus):
    return ' '.join([document.strip() 
                     for document in corpus])
                         
def compute_ngrams(sequence, n):
    return zip(*[sequence[index:] 
                 for index in range(n)])


def get_top_ngrams(corpus, ngram_val=1, limit=5):

    corpus = flatten_corpus(corpus)
    tokens = nltk.word_tokenize(corpus)

    ngrams = compute_ngrams(tokens, ngram_val)
    ngrams_freq_dist = nltk.FreqDist(ngrams)
    sorted_ngrams_fd = sorted(ngrams_freq_dist.items(), 
                              key=itemgetter(1), reverse=True)
    sorted_ngrams = sorted_ngrams_fd[0:limit]
    sorted_ngrams = [(' '.join(text), freq) 
                     for text, freq in sorted_ngrams]

    return sorted_ngrams   
    

get_top_ngrams(corpus=norm_alice, ngram_val=2,
               limit=10)
               
get_top_ngrams(corpus=norm_alice, ngram_val=3,
               limit=10)

from nltk.collocations import BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures

finder = BigramCollocationFinder.from_documents([item.split() 
                                                for item 
                                                in norm_alice])
bigram_measures = BigramAssocMeasures()                                                
finder.nbest(bigram_measures.raw_freq, 10)
finder.nbest(bigram_measures.pmi, 10)   

from nltk.collocations import TrigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures

finder = TrigramCollocationFinder.from_documents([item.split() 
                                                for item 
                                                in norm_alice])
trigram_measures = TrigramAssocMeasures()                                                
finder.nbest(trigram_measures.raw_freq, 10)
finder.nbest(trigram_measures.pmi, 10)  


toy_text = """
Elephants are large mammals of the family Elephantidae 
and the order Proboscidea. Two species are traditionally recognised, 
the African elephant and the Asian elephant. Elephants are scattered 
throughout sub-Saharan Africa, South Asia, and Southeast Asia. Male 
African elephants are the largest extant terrestrial animals. All 
elephants have a long trunk used for many purposes, 
particularly breathing, lifting water and grasping objects. Their 
incisors grow into tusks, which can serve as weapons and as tools 
for moving objects and digging. Elephants' large ear flaps help 
to control their body temperature. Their pillar-like legs can 
carry their great weight. African elephants have larger ears 
and concave backs while Asian elephants have smaller ears 
and convex or level backs.  
"""

from normalization import parse_document
import itertools
import nltk
from normalization import stopword_list
from gensim import corpora, models


def get_chunks(sentences, grammar = r'NP: {<DT>? <JJ>* <NN.*>+}'):
    
    all_chunks = []
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    
    for sentence in sentences:
        
        tagged_sents = nltk.pos_tag_sents(
                            [nltk.word_tokenize(sentence)])
        
        chunks = [chunker.parse(tagged_sent) 
                  for tagged_sent in tagged_sents]
        
        wtc_sents = [nltk.chunk.tree2conlltags(chunk)
                     for chunk in chunks]    
         
        flattened_chunks = list(
                            itertools.chain.from_iterable(
                                wtc_sent for wtc_sent in wtc_sents)
                           )
        
        valid_chunks_tagged = [(status, [wtc for wtc in chunk]) 
                        for status, chunk 
                        in itertools.groupby(flattened_chunks, 
                                             lambda (word,pos,chunk): chunk != 'O')]
        
        valid_chunks = [' '.join(word.lower() 
                                for word, tag, chunk 
                                in wtc_group 
                                    if word.lower() 
                                        not in stopword_list) 
                                    for status, wtc_group 
                                    in valid_chunks_tagged
                                        if status]
                                            
        all_chunks.append(valid_chunks)
    
    return all_chunks
    
sentences = parse_document(toy_text)          
valid_chunks = get_chunks(sentences)
print valid_chunks

def get_tfidf_weighted_keyphrases(sentences, 
                                  grammar=r'NP: {<DT>? <JJ>* <NN.*>+}',
                                  top_n=10):
    
    valid_chunks = get_chunks(sentences, grammar=grammar)
                                     
    dictionary = corpora.Dictionary(valid_chunks)
    corpus = [dictionary.doc2bow(chunk) for chunk in valid_chunks]
    
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    
    weighted_phrases = {dictionary.get(id): round(value,3) 
                        for doc in corpus_tfidf 
                        for id, value in doc}
                            
    weighted_phrases = sorted(weighted_phrases.items(), 
                              key=itemgetter(1), reverse=True)
    
    return weighted_phrases[:top_n]
    
get_tfidf_weighted_keyphrases(sentences, top_n=10)

# try on other corpora!
get_tfidf_weighted_keyphrases(alice, top_n=10)
    
    
    

