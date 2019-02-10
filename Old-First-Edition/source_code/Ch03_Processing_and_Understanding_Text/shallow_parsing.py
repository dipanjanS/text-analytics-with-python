# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 22:22:02 2016

@author: DIP
"""

from pattern.en import parsetree, Chunk
from nltk.tree import Tree

sentence = 'The brown fox is quick and he is jumping over the lazy dog'

tree = parsetree(sentence)
print tree

for sentence_tree in tree:
    print sentence_tree.chunks
    
for sentence_tree in tree:
    for chunk in sentence_tree.chunks:
        print chunk.type, '->', [(word.string, word.type) 
                                 for word in chunk.words]
        

def create_sentence_tree(sentence, lemmatize=False):
    sentence_tree = parsetree(sentence, 
                              relations=True, 
                              lemmata=lemmatize)
    return sentence_tree[0]
    
def get_sentence_tree_constituents(sentence_tree):
    return sentence_tree.constituents()
    
def process_sentence_tree(sentence_tree):
    
    tree_constituents = get_sentence_tree_constituents(sentence_tree)
    processed_tree = [
                        (item.type,
                         [
                             (w.string, w.type)
                             for w in item.words
                         ]
                        )
                        if type(item) == Chunk
                        else ('-',
                              [
                                   (item.string, item.type)
                              ]
                             )
                             for item in tree_constituents
                    ]
    
    return processed_tree
    
def print_sentence_tree(sentence_tree):
    

    processed_tree = process_sentence_tree(sentence_tree)
    processed_tree = [
                        Tree( item[0],
                             [
                                 Tree(x[1], [x[0]])
                                 for x in item[1]
                             ]
                            )
                            for item in processed_tree
                     ]

    tree = Tree('S', processed_tree )
    print tree
    
def visualize_sentence_tree(sentence_tree):
    
    processed_tree = process_sentence_tree(sentence_tree)
    processed_tree = [
                        Tree( item[0],
                             [
                                 Tree(x[1], [x[0]])
                                 for x in item[1]
                             ]
                            )
                            for item in processed_tree
                     ]
    tree = Tree('S', processed_tree )
    tree.draw()
    
    
t = create_sentence_tree(sentence)
print t

pt = process_sentence_tree(t)
pt

print_sentence_tree(t)
visualize_sentence_tree(t)



from nltk.corpus import treebank_chunk
data = treebank_chunk.chunked_sents()
train_data = data[:4000]
test_data = data[4000:]
print train_data[7]

simple_sentence = 'the quick fox jumped over the lazy dog'

from nltk.chunk import RegexpParser
from pattern.en import tag

tagged_simple_sent = tag(simple_sentence)
print tagged_simple_sent

chunk_grammar = """
NP: {<DT>?<JJ>*<NN.*>}
"""
rc = RegexpParser(chunk_grammar)
c = rc.parse(tagged_simple_sent)
print c

chink_grammar = """
NP: {<.*>+} # chunk everything as NP
}<VBD|IN>+{
"""
rc = RegexpParser(chink_grammar)
c = rc.parse(tagged_simple_sent)
print c

tagged_sentence = tag(sentence)
print tagged_sentence

grammar = """
NP: {<DT>?<JJ>?<NN.*>}  
ADJP: {<JJ>}
ADVP: {<RB.*>}
PP: {<IN>}      
VP: {<MD>?<VB.*>+}

"""
rc = RegexpParser(grammar)
c = rc.parse(tagged_sentence)

print c

print rc.evaluate(test_data)


   


from nltk.chunk.util import tree2conlltags, conlltags2tree

train_sent = train_data[7]
print train_sent

wtc = tree2conlltags(train_sent)
wtc

tree = conlltags2tree(wtc)
print tree
    

def conll_tag_chunks(chunk_sents):
  tagged_sents = [tree2conlltags(tree) for tree in chunk_sents]
  return [[(t, c) for (w, t, c) in sent] for sent in tagged_sents]
  
def combined_tagger(train_data, taggers, backoff=None):
    for tagger in taggers:
        backoff = tagger(train_data, backoff=backoff)
    return backoff
  
from nltk.tag import UnigramTagger, BigramTagger
from nltk.chunk import ChunkParserI

class NGramTagChunker(ChunkParserI):
    
  def __init__(self, train_sentences, 
               tagger_classes=[UnigramTagger, BigramTagger]):
    train_sent_tags = conll_tag_chunks(train_sentences)
    self.chunk_tagger = combined_tagger(train_sent_tags, tagger_classes)

  def parse(self, tagged_sentence):
    if not tagged_sentence: 
        return None
    pos_tags = [tag for word, tag in tagged_sentence]
    chunk_pos_tags = self.chunk_tagger.tag(pos_tags)
    chunk_tags = [chunk_tag for (pos_tag, chunk_tag) in chunk_pos_tags]
    wpc_tags = [(word, pos_tag, chunk_tag) for ((word, pos_tag), chunk_tag)
                     in zip(tagged_sentence, chunk_tags)]
    return conlltags2tree(wpc_tags)


ntc = NGramTagChunker(train_data)
print ntc.evaluate(test_data)

tree = ntc.parse(tagged_sentence)
print tree
tree.draw()


from nltk.corpus import conll2000
wsj_data = conll2000.chunked_sents()
train_wsj_data = wsj_data[:7500]
test_wsj_data = wsj_data[7500:]
print train_wsj_data[10]

tc = NGramTagChunker(train_wsj_data)
print tc.evaluate(test_wsj_data)