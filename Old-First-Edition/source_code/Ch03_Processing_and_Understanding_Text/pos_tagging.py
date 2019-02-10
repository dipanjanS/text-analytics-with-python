# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 17:08:05 2016

@author: DIP
"""

sentence = 'The brown fox is quick and he is jumping over the lazy dog'


# recommended tagger based on PTB
import nltk
tokens = nltk.word_tokenize(sentence)
tagged_sent = nltk.pos_tag(tokens, tagset='universal')
print tagged_sent

from pattern.en import tag
tagged_sent = tag(sentence)
print tagged_sent


# building your own tagger

# preparing the data
from nltk.corpus import treebank
data = treebank.tagged_sents()
train_data = data[:3500]
test_data = data[3500:]
print train_data[0]

# default tagger
from nltk.tag import DefaultTagger
dt = DefaultTagger('NN')

print dt.evaluate(test_data)

print dt.tag(tokens)


# regex tagger
from nltk.tag import RegexpTagger
# define regex tag patterns
patterns = [
        (r'.*ing$', 'VBG'),               # gerunds
        (r'.*ed$', 'VBD'),                # simple past
        (r'.*es$', 'VBZ'),                # 3rd singular present
        (r'.*ould$', 'MD'),               # modals
        (r'.*\'s$', 'NN$'),               # possessive nouns
        (r'.*s$', 'NNS'),                 # plural nouns
        (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
        (r'.*', 'NN')                     # nouns (default) ... 
]
rt = RegexpTagger(patterns)

print rt.evaluate(test_data)
print rt.tag(tokens)


## N gram taggers
from nltk.tag import UnigramTagger
from nltk.tag import BigramTagger
from nltk.tag import TrigramTagger

ut = UnigramTagger(train_data)
bt = BigramTagger(train_data)
tt = TrigramTagger(train_data)

print ut.evaluate(test_data)
print ut.tag(tokens)

print bt.evaluate(test_data)
print bt.tag(tokens)

print tt.evaluate(test_data)
print tt.tag(tokens)

def combined_tagger(train_data, taggers, backoff=None):
    for tagger in taggers:
        backoff = tagger(train_data, backoff=backoff)
    return backoff

ct = combined_tagger(train_data=train_data, 
                     taggers=[UnigramTagger, BigramTagger, TrigramTagger],
                     backoff=rt)

print ct.evaluate(test_data)        
print ct.tag(tokens)

from nltk.classify import NaiveBayesClassifier, MaxentClassifier
from nltk.tag.sequential import ClassifierBasedPOSTagger

nbt = ClassifierBasedPOSTagger(train=train_data,
                               classifier_builder=NaiveBayesClassifier.train)

print nbt.evaluate(test_data)
print nbt.tag(tokens)    


# try this out for fun!
met = ClassifierBasedPOSTagger(train=train_data,
                               classifier_builder=MaxentClassifier.train)
print met.evaluate(test_data)                           
print met.tag(tokens)
