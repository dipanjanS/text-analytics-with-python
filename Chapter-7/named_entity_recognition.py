# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 21:06:57 2016

@author: DIP
"""

text = """
Bayern Munich, or FC Bayern, is a German sports club based in Munich, 
Bavaria, Germany. It is best known for its professional football team, 
which plays in the Bundesliga, the top tier of the German football 
league system, and is the most successful club in German football 
history, having won a record 26 national titles and 18 national cups. 
FC Bayern was founded in 1900 by eleven football players led by Franz John. 
Although Bayern won its first national championship in 1932, the club 
was not selected for the Bundesliga at its inception in 1963. The club 
had its period of greatest success in the middle of the 1970s when, 
under the captaincy of Franz Beckenbauer, it won the European Cup three 
times in a row (1974-76). Overall, Bayern has reached ten UEFA Champions 
League finals, most recently winning their fifth title in 2013 as part 
of a continental treble. 
"""

import nltk
from normalization import parse_document
import pandas as pd

sentences = parse_document(text)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]


# nltk NER
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
ne_chunked_sents = [nltk.ne_chunk(tagged) for tagged in tagged_sentences]
named_entities = []
for ne_tagged_sentence in ne_chunked_sents:
    for tagged_tree in ne_tagged_sentence:
        if hasattr(tagged_tree, 'label'):
                entity_name = ' '.join(c[0] for c in tagged_tree.leaves())
                entity_type = tagged_tree.label()
                named_entities.append((entity_name, entity_type))
                
named_entities = list(set(named_entities))
entity_frame = pd.DataFrame(named_entities, 
                            columns=['Entity Name', 'Entity Type'])
print entity_frame    



# stanford NER
from nltk.tag import StanfordNERTagger
import os
java_path = r'C:\Program Files\Java\jdk1.8.0_102\bin\java.exe'
os.environ['JAVAHOME'] = java_path

sn = StanfordNERTagger('E:/stanford/stanford-ner-2014-08-27/classifiers/english.all.3class.distsim.crf.ser.gz',
                       path_to_jar='E:/stanford/stanford-ner-2014-08-27/stanford-ner.jar')
                       
ne_annotated_sentences = [sn.tag(sent) for sent in tokenized_sentences]

named_entities = []
for sentence in ne_annotated_sentences:
    temp_entity_name = ''
    temp_named_entity = None
    for term, tag in sentence:
        if tag != 'O':
            temp_entity_name = ' '.join([temp_entity_name, term]).strip()
            temp_named_entity = (temp_entity_name, tag)
        else:
            if temp_named_entity:
                named_entities.append(temp_named_entity)
                temp_entity_name = ''
                temp_named_entity = None

named_entities = list(set(named_entities))
entity_frame = pd.DataFrame(named_entities, 
                            columns=['Entity Name', 'Entity Type'])
print entity_frame                       