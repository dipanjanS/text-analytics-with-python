# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 06:55:49 2016

@author: DIP
"""

sentence = 'The brown fox is quick and he is jumping over the lazy dog'

# set java path
import os
java_path = r'C:\Program Files\Java\jdk1.8.0_102\bin\java.exe'
os.environ['JAVAHOME'] = java_path

from nltk.parse.stanford import StanfordParser

scp = StanfordParser(path_to_jar='E:/stanford/stanford-parser-full-2015-04-20/stanford-parser.jar',
                   path_to_models_jar='E:/stanford/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models.jar')
                   
result = list(scp.raw_parse(sentence))
print result[0]

result[0].draw()

import nltk
from nltk.grammar import Nonterminal
from nltk.corpus import treebank

training_set = treebank.parsed_sents()

print training_set[1]

# extract the productions for all annotated training sentences
treebank_productions = list(
                        set(production 
                            for sent in training_set  
                            for production in sent.productions()
                        )
                    )

treebank_productions[0:10]
  
# add productions for each word, POS tag
for word, tag in treebank.tagged_words():
	t = nltk.Tree.fromstring("("+ tag + " " + word  +")")
	for production in t.productions():
		treebank_productions.append(production)

# build the PCFG based grammar  
treebank_grammar = nltk.grammar.induce_pcfg(Nonterminal('S'), 
                                         treebank_productions)

# build the parser
viterbi_parser = nltk.ViterbiParser(treebank_grammar)

# get sample sentence tokens
tokens = nltk.word_tokenize(sentence)

# get parse tree for sample sentence
result = list(viterbi_parser.parse(tokens))


# get tokens and their POS tags
from pattern.en import tag as pos_tagger
tagged_sent = pos_tagger(sentence)

print tagged_sent

# extend productions for sample sentence tokens
for word, tag in tagged_sent:
    t = nltk.Tree.fromstring("("+ tag + " " + word  +")")
    for production in t.productions():
		treebank_productions.append(production)

# rebuild grammar
treebank_grammar = nltk.grammar.induce_pcfg(Nonterminal('S'), 
                                         treebank_productions)                                         

# rebuild parser
viterbi_parser = nltk.ViterbiParser(treebank_grammar)

# get parse tree for sample sentence
result = list(viterbi_parser.parse(tokens))

print result[0]
result[0].draw()                  