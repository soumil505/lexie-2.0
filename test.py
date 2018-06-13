# -*- coding: utf-8 -*-
"""
Created on Sat May 26 08:25:51 2018

@author: soumil
"""

from model2 import architecture
import scraper2
from preprocessing import *
import tensorflow as tf

allowed_chars="qwertyuiopasdfghjklzxcvbnm'-_1234567890 "
stopwords=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "don't", "should", "now"]
links=["http://beehex.com/"]
sequence_length=30
'''
a=scraper2.links_to_text(links)
b=generate_word_matrix_pairs(a,allowed_chars,stopwords,grouped_words=2,sequence_length=sequence_length)
c=generate_word_pairs(a,allowed_chars,stopwords,grouped_words=2)
'''
w1,w2,target,output,loss=architecture(allowed_chars,sequence_length,16)
optimizer = tf.train.AdamOptimizer().minimize(loss)