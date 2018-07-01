# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import nltk
from nltk.corpus import gutenberg
from pprint import pprint

## SENTENCE TOKENIZATION

# loading text corpora
alice = gutenberg.raw(fileids='carroll-alice.txt')
sample_text = 'We will discuss briefly about the basic syntax,\
 structure and design philosophies. \
 There is a defined hierarchical syntax for Python code which you should remember \
 when writing code! Python is a really powerful programming language!'
               
# Total characters in Alice in Wonderland
print len(alice)
# First 100 characters in the corpus
print alice[0:100]
print
                
## default sentence tokenizer
default_st = nltk.sent_tokenize
alice_sentences = default_st(text=alice)
sample_sentences = default_st(text=sample_text)

print 'Total sentences in sample_text:', len(sample_sentences)
print 'Sample text sentences :-'
pprint(sample_sentences)
print '\nTotal sentences in alice:', len(alice_sentences)
print 'First 5 sentences in alice:-'
pprint(alice_sentences[0:5])


## Other languages sentence tokenization
from nltk.corpus import europarl_raw

german_text = europarl_raw.german.raw(fileids='ep-00-01-17.de')
# Total characters in the corpus
print len(german_text)
# First 100 characters in the corpus
print german_text[0:100]
print

# default sentence tokenizer 
german_sentences_def = default_st(text=german_text, language='german')

# loading german text tokenizer into a PunktSentenceTokenizer instance  
german_tokenizer = nltk.data.load(resource_url='tokenizers/punkt/german.pickle')
german_sentences = german_tokenizer.tokenize(german_text)

# verify the type of german_tokenizer
# should be PunktSentenceTokenizer
print type(german_tokenizer)

# check if results of both tokenizers match
# should be True
print german_sentences_def == german_sentences
# print first 5 sentences of the corpus
for sent in german_sentences[0:5]:
    print sent


## using PunktSentenceTokenizer for sentence tokenization
punkt_st = nltk.tokenize.PunktSentenceTokenizer()
sample_sentences = punkt_st.tokenize(sample_text)
pprint(sample_sentences)

## using RegexpTokenizer for sentence tokenization
SENTENCE_TOKENS_PATTERN = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s'
regex_st = nltk.tokenize.RegexpTokenizer(
            pattern=SENTENCE_TOKENS_PATTERN,
            gaps=True)
sample_sentences = regex_st.tokenize(sample_text)
pprint(sample_sentences)         
        
## WORD TOKENIZATION

sentence = "The brown fox wasn't that quick and he couldn't win the race"

# default word tokenizer
default_wt = nltk.word_tokenize
words = default_wt(sentence)
print words       

# treebank word tokenizer
treebank_wt = nltk.TreebankWordTokenizer()
words = treebank_wt.tokenize(sentence)
print words
        
# regex word tokenizer
TOKEN_PATTERN = r'\w+'        
regex_wt = nltk.RegexpTokenizer(pattern=TOKEN_PATTERN,
                                gaps=False)
words = regex_wt.tokenize(sentence)
print words

GAP_PATTERN = r'\s+'        
regex_wt = nltk.RegexpTokenizer(pattern=GAP_PATTERN,
                                gaps=True)
words = regex_wt.tokenize(sentence)
print words

word_indices = list(regex_wt.span_tokenize(sentence))
print word_indices
print [sentence[start:end] for start, end in word_indices]

# derived regex tokenizers
wordpunkt_wt = nltk.WordPunctTokenizer()
words = wordpunkt_wt.tokenize(sentence)
print words

whitespace_wt = nltk.WhitespaceTokenizer()
words = whitespace_wt.tokenize(sentence)
print words