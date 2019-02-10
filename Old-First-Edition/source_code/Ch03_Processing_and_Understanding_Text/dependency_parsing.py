# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 04:25:12 2016

@author: DIP
"""


sentence = 'The brown fox is quick and he is jumping over the lazy dog'

from spacy.en import English
parser = English()
parsed_sent = parser(unicode(sentence))

dependency_pattern = '{left}<---{word}[{w_type}]--->{right}\n--------'
for token in parsed_sent:
    print dependency_pattern.format(word=token.orth_, 
                                  w_type=token.dep_,
                                  left=[t.orth_ 
                                            for t 
                                            in token.lefts],
                                  right=[t.orth_ 
                                             for t 
                                             in token.rights])
                                             

# set java path
import os
java_path = r'C:\Program Files\Java\jdk1.8.0_102\bin\java.exe'
os.environ['JAVAHOME'] = java_path
                                             
from nltk.parse.stanford import StanfordDependencyParser
sdp = StanfordDependencyParser(path_to_jar='E:/stanford/stanford-parser-full-2015-04-20/stanford-parser.jar',
                               path_to_models_jar='E:/stanford/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models.jar')    
result = list(sdp.raw_parse(sentence))  

result[0]

[item for item in result[0].triples()]

dep_tree = [parse.tree() for parse in result][0]
print dep_tree
dep_tree.draw()
                           
                          
import nltk
tokens = nltk.word_tokenize(sentence)

dependency_rules = """
'fox' -> 'The' | 'brown'
'quick' -> 'fox' | 'is' | 'and' | 'jumping'
'jumping' -> 'he' | 'is' | 'dog'
'dog' -> 'over' | 'the' | 'lazy'
"""

dependency_grammar = nltk.grammar.DependencyGrammar.fromstring(dependency_rules)
print dependency_grammar

dp = nltk.ProjectiveDependencyParser(dependency_grammar)
res = [item for item in dp.parse(tokens)]
tree = res[0] 
print tree

tree.draw()                          