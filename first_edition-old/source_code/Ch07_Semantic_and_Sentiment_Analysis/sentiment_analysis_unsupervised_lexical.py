# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 19:45:40 2016

@author: DIP
"""

import pandas as pd
import numpy as np



dataset = pd.read_csv(r'E:/aclImdb/movie_reviews.csv')

print dataset.head()

train_data = dataset[:35000]
test_data = dataset[35000:]

test_reviews = np.array(test_data['review'])
test_sentiments = np.array(test_data['sentiment'])


sample_docs = [100, 5817, 7626, 7356, 1008, 7155, 3533, 13010]
sample_data = [(test_reviews[index],
                test_sentiments[index])
                  for index in sample_docs]


sample_data        


from afinn import Afinn
afn = Afinn(emoticons=True) 
print afn.score('I really hated the plot of this movie')

print afn.score('I really hated the plot of this movie :(')



import nltk
from nltk.corpus import sentiwordnet as swn

good = swn.senti_synsets('good', 'n')[0]
print 'Positive Polarity Score:', good.pos_score()
print 'Negative Polarity Score:', good.neg_score()
print 'Objective Score:', good.obj_score()

from normalization import normalize_accented_characters, html_parser, strip_html

def analyze_sentiment_sentiwordnet_lexicon(review,
                                           verbose=False):
    # pre-process text
    review = normalize_accented_characters(review)
    review = html_parser.unescape(review)
    review = strip_html(review)
    # tokenize and POS tag text tokens
    text_tokens = nltk.word_tokenize(review)
    tagged_text = nltk.pos_tag(text_tokens)
    pos_score = neg_score = token_count = obj_score = 0
    # get wordnet synsets based on POS tags
    # get sentiment scores if synsets are found
    for word, tag in tagged_text:
        ss_set = None
        if 'NN' in tag and swn.senti_synsets(word, 'n'):
            ss_set = swn.senti_synsets(word, 'n')[0]
        elif 'VB' in tag and swn.senti_synsets(word, 'v'):
            ss_set = swn.senti_synsets(word, 'v')[0]
        elif 'JJ' in tag and swn.senti_synsets(word, 'a'):
            ss_set = swn.senti_synsets(word, 'a')[0]
        elif 'RB' in tag and swn.senti_synsets(word, 'r'):
            ss_set = swn.senti_synsets(word, 'r')[0]
        # if senti-synset is found        
        if ss_set:
            # add scores for all found synsets
            pos_score += ss_set.pos_score()
            neg_score += ss_set.neg_score()
            obj_score += ss_set.obj_score()
            token_count += 1
    
    # aggregate final scores
    final_score = pos_score - neg_score
    norm_final_score = round(float(final_score) / token_count, 2)
    final_sentiment = 'positive' if norm_final_score >= 0 else 'negative'
    if verbose:
        norm_obj_score = round(float(obj_score) / token_count, 2)
        norm_pos_score = round(float(pos_score) / token_count, 2)
        norm_neg_score = round(float(neg_score) / token_count, 2)
        # to display results in a nice table
        sentiment_frame = pd.DataFrame([[final_sentiment, norm_obj_score,
                                         norm_pos_score, norm_neg_score,
                                         norm_final_score]],
                                         columns=pd.MultiIndex(levels=[['SENTIMENT STATS:'], 
                                                                      ['Predicted Sentiment', 'Objectivity',
                                                                       'Positive', 'Negative', 'Overall']], 
                                                              labels=[[0,0,0,0,0],[0,1,2,3,4]]))
        print sentiment_frame
        
    return final_sentiment
            
            
for review, review_sentiment in sample_data:  
    print 'Review:'
    print review
    print
    print 'Labeled Sentiment:', review_sentiment    
    print    
    final_sentiment = analyze_sentiment_sentiwordnet_lexicon(review,
                                                             verbose=True)
    print '-'*60                                                         

sentiwordnet_predictions = [analyze_sentiment_sentiwordnet_lexicon(review)
                            for review in test_reviews]

from utils import display_evaluation_metrics, display_confusion_matrix, display_classification_report

print 'Performance metrics:'
display_evaluation_metrics(true_labels=test_sentiments,
                           predicted_labels=sentiwordnet_predictions,
                           positive_class='positive')  
print '\nConfusion Matrix:'                           
display_confusion_matrix(true_labels=test_sentiments,
                         predicted_labels=sentiwordnet_predictions,
                         classes=['positive', 'negative'])
print '\nClassification report:'                         
display_classification_report(true_labels=test_sentiments,
                              predicted_labels=sentiwordnet_predictions,
                              classes=['positive', 'negative'])  

                                                

from nltk.sentiment.vader import SentimentIntensityAnalyzer

def analyze_sentiment_vader_lexicon(review, 
                                    threshold=0.1,
                                    verbose=False):
    # pre-process text
    review = normalize_accented_characters(review)
    review = html_parser.unescape(review)
    review = strip_html(review)
    # analyze the sentiment for review
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(review)
    # get aggregate scores and final sentiment
    agg_score = scores['compound']
    final_sentiment = 'positive' if agg_score >= threshold\
                                   else 'negative'
    if verbose:
        # display detailed sentiment statistics
        positive = str(round(scores['pos'], 2)*100)+'%'
        final = round(agg_score, 2)
        negative = str(round(scores['neg'], 2)*100)+'%'
        neutral = str(round(scores['neu'], 2)*100)+'%'
        sentiment_frame = pd.DataFrame([[final_sentiment, final, positive,
                                        negative, neutral]],
                                        columns=pd.MultiIndex(levels=[['SENTIMENT STATS:'], 
                                                                      ['Predicted Sentiment', 'Polarity Score',
                                                                       'Positive', 'Negative',
                                                                       'Neutral']], 
                                                              labels=[[0,0,0,0,0],[0,1,2,3,4]]))
        print sentiment_frame
    
    return final_sentiment
        
    
    

for review, review_sentiment in sample_data:
    print 'Review:'
    print review
    print
    print 'Labeled Sentiment:', review_sentiment    
    print    
    final_sentiment = analyze_sentiment_vader_lexicon(review,
                                                        threshold=0.1,
                                                        verbose=True)
    print '-'*60                                                       

vader_predictions = [analyze_sentiment_vader_lexicon(review, threshold=0.1)
                     for review in test_reviews] 

print 'Performance metrics:'
display_evaluation_metrics(true_labels=test_sentiments,
                           predicted_labels=vader_predictions,
                           positive_class='positive')  
print '\nConfusion Matrix:'                           
display_confusion_matrix(true_labels=test_sentiments,
                         predicted_labels=vader_predictions,
                         classes=['positive', 'negative'])
print '\nClassification report:'                         
display_classification_report(true_labels=test_sentiments,
                              predicted_labels=vader_predictions,
                              classes=['positive', 'negative']) 

  

from pattern.en import sentiment, mood, modality

def analyze_sentiment_pattern_lexicon(review, threshold=0.1,
                                      verbose=False):
    # pre-process text
    review = normalize_accented_characters(review)
    review = html_parser.unescape(review)
    review = strip_html(review)
    # analyze sentiment for the text document
    analysis = sentiment(review)
    sentiment_score = round(analysis[0], 2)
    sentiment_subjectivity = round(analysis[1], 2)
    # get final sentiment
    final_sentiment = 'positive' if sentiment_score >= threshold\
                                   else 'negative'
    if verbose:
        # display detailed sentiment statistics
        sentiment_frame = pd.DataFrame([[final_sentiment, sentiment_score,
                                        sentiment_subjectivity]],
                                        columns=pd.MultiIndex(levels=[['SENTIMENT STATS:'], 
                                                                      ['Predicted Sentiment', 'Polarity Score',
                                                                       'Subjectivity Score']], 
                                                              labels=[[0,0,0],[0,1,2]]))
        print sentiment_frame
        assessment = analysis.assessments
        assessment_frame = pd.DataFrame(assessment, 
                                        columns=pd.MultiIndex(levels=[['DETAILED ASSESSMENT STATS:'], 
                                                                      ['Key Terms', 'Polarity Score',
                                                                       'Subjectivity Score', 'Type']], 
                                                              labels=[[0,0,0,0],[0,1,2,3]]))
        print assessment_frame
        print
    
    return final_sentiment                                       
    

for review, review_sentiment in sample_data:
    print 'Review:'
    print review
    print
    print 'Labeled Sentiment:', review_sentiment    
    print    
    final_sentiment = analyze_sentiment_pattern_lexicon(review,
                                                        threshold=0.1,
                                                        verbose=True)
    print '-'*60            
      
for review, review_sentiment in sample_data:
    print 'Review:'
    print review
    print 'Labeled Sentiment:', review_sentiment 
    print 'Mood:', mood(review)
    mod_score = modality(review)
    print 'Modality Score:', round(mod_score, 2)
    print 'Certainty:', 'Strong' if mod_score > 0.5 \
                                    else 'Medium' if mod_score > 0.35 \
                                                    else 'Low'
    print '-'*60            

                  




                               
                                
pattern_predictions = [analyze_sentiment_pattern_lexicon(review, threshold=0.1)
                            for review in test_reviews]     

                              
print 'Performance metrics:'
display_evaluation_metrics(true_labels=test_sentiments,
                           predicted_labels=pattern_predictions,
                           positive_class='positive')  
print '\nConfusion Matrix:'                           
display_confusion_matrix(true_labels=test_sentiments,
                         predicted_labels=pattern_predictions,
                         classes=['positive', 'negative'])
print '\nClassification report:'                         
display_classification_report(true_labels=test_sentiments,
                              predicted_labels=pattern_predictions,
                              classes=['positive', 'negative'])                               
