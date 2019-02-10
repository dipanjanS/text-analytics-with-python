# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 23:21:51 2016

@author: DIP
"""

import nltk
import re
import string
from pprint import pprint

corpus = ["The brown fox wasn't that quick and he couldn't win the race",
          "Hey that's a great deal! I just bought a phone for $199",
          "@@You'll (learn) a **lot** in the book. Python is an amazing language!@@"]


def tokenize_text(text):
    sentences = nltk.sent_tokenize(text)
    word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences] 
    return word_tokens
    
token_list = [tokenize_text(text) 
              for text in corpus]
pprint(token_list)
print


    
def remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    return filtered_tokens
    
filtered_list_1 =  [filter(None,[remove_characters_after_tokenization(tokens) 
                                for tokens in sentence_tokens]) 
                    for sentence_tokens in token_list]
print filtered_list_1
print  


def remove_characters_before_tokenization(sentence,
                                          keep_apostrophes=False):
    sentence = sentence.strip()
    if keep_apostrophes:
        PATTERN = r'[?|$|&|*|%|@|(|)|~]'
        filtered_sentence = re.sub(PATTERN, r'', sentence)
    else:
        PATTERN = r'[^a-zA-Z0-9 ]'
        filtered_sentence = re.sub(PATTERN, r'', sentence)
    return filtered_sentence
    
filtered_list_2 = [remove_characters_before_tokenization(sentence) 
                    for sentence in corpus]    
print filtered_list_2
print 

cleaned_corpus = [remove_characters_before_tokenization(sentence, keep_apostrophes=True) 
                  for sentence in corpus]
print cleaned_corpus


from contractions import CONTRACTION_MAP

def expand_contractions(sentence, contraction_mapping):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_sentence = contractions_pattern.sub(expand_match, sentence)
    return expanded_sentence
    
expanded_corpus = [expand_contractions(sentence, CONTRACTION_MAP) 
                    for sentence in cleaned_corpus]    
print expanded_corpus
print 

    
# case conversion    
print corpus[0].lower()
print corpus[0].upper()
 
       
# removing stopwords
def remove_stopwords(tokens):
    stopword_list = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens
    
expanded_corpus_tokens = [tokenize_text(text)
                          for text in expanded_corpus]    
filtered_list_3 =  [[remove_stopwords(tokens) 
                        for tokens in sentence_tokens] 
                        for sentence_tokens in expanded_corpus_tokens]
print filtered_list_3
print 


# removing repeated characters
sample_sentence = 'My schooool is realllllyyy amaaazingggg'
sample_sentence_tokens = tokenize_text(sample_sentence)[0]

from nltk.corpus import wordnet

def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'
    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word
            
    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens

print remove_repeated_characters(sample_sentence_tokens)    


# porter stemmer
from nltk.stem import PorterStemmer
ps = PorterStemmer()

print ps.stem('jumping'), ps.stem('jumps'), ps.stem('jumped')

print ps.stem('lying')

print ps.stem('strange')

# lancaster stemmer
from nltk.stem import LancasterStemmer
ls = LancasterStemmer()

print ls.stem('jumping'), ls.stem('jumps'), ls.stem('jumped')

print ls.stem('lying')

print ls.stem('strange')


# regex stemmer
from nltk.stem import RegexpStemmer
rs = RegexpStemmer('ing$|s$|ed$', min=4)

print rs.stem('jumping'), rs.stem('jumps'), rs.stem('jumped')

print rs.stem('lying')

print rs.stem('strange')


# snowball stemmer
from nltk.stem import SnowballStemmer
ss = SnowballStemmer("german")

print 'Supported Languages:', SnowballStemmer.languages

# autobahnen -> cars
# autobahn -> car
ss.stem('autobahnen')

# springen -> jumping
# spring -> jump
ss.stem('springen')


# lemmatization
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

# lemmatize nouns
print wnl.lemmatize('cars', 'n')
print wnl.lemmatize('men', 'n')

# lemmatize verbs
print wnl.lemmatize('running', 'v')
print wnl.lemmatize('ate', 'v')

# lemmatize adjectives
print wnl.lemmatize('saddest', 'a')
print wnl.lemmatize('fancier', 'a')

# ineffective lemmatization
print wnl.lemmatize('ate', 'n')
print wnl.lemmatize('fancier', 'v')
