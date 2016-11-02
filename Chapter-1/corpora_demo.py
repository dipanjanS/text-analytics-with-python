# -*- coding: utf-8 -*-
"""
Created on Tue Oct 04 23:08:05 2016

@author: DIP
"""

# BROWN CORPUS DEMO
from nltk.corpus import brown
import nltk

print 'Total Categories:', len(brown.categories())

print brown.categories()

# tokenized sentences
brown.sents(categories='mystery')

# POS tagged sentences
brown.tagged_sents(categories='mystery')

# get sentences in natural form
sentences = brown.sents(categories='mystery')

# get tagged words
tagged_words = brown.tagged_words(categories='mystery')

# get nouns from tagged words
nouns = [(word, tag) for word, tag in tagged_words if any(noun_tag in tag for noun_tag in ['NP', 'NN'])]

print nouns[0:10] # prints the first 10 nouns

# build frequency distribution for nouns
nouns_freq = nltk.FreqDist([word for word, tag in nouns])

# print top 10 occuring nouns
print nouns_freq.most_common(10)


# REUTERS CORPUS DEMO
from nltk.corpus import reuters

print 'Total Categories:', len(reuters.categories())

print reuters.categories()

# get sentences in housing and income categories
sentences = reuters.sents(categories=['housing', 'income'])
sentences = [' '.join(sentence_tokens) for sentence_tokens in sentences]

print sentences[0:5]  # prints the first 5 sentences

# fileid based access
print reuters.fileids(categories=['housing', 'income'])

print reuters.sents(fileids=[u'test/16118', u'test/18534'])


# WORDNET CORPUS DEMO
from nltk.corpus import wordnet as wn

word = 'hike' # taking hike as our word of interest

# get word synsets
word_synsets = wn.synsets(word)
print word_synsets

# get details for each synonym in synset
for synset in word_synsets:
    print 'Synset Name:', synset.name()
    print 'POS Tag:', synset.pos()
    print 'Definition:', synset.definition()
    print 'Examples:', synset.examples()
    print
    





