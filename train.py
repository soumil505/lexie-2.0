# -*- coding: utf-8 -*-
"""
Created on Mon May 28 22:21:48 2018

@author: soumil
"""

import tensorflow as tf
from model2 import architecture
import scraper2
from preprocessing import generate_word_matrix_pairs,word2mat
import random
import numpy as np

allowed_chars="qwertyuiopasdfghjklzxcvbnm'-_1234567890 "
stopwords=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "don't", "should", "now"]
links=["http://beehex.com/"]
sequence_length=30

text=scraper2.links_to_text(links)
word_pairs=generate_word_matrix_pairs(text,allowed_chars,stopwords,grouped_words=1,sequence_length=sequence_length)


def generate_batches(seq,num_batches=30,seed=1):
    random.seed(seed)
    random.shuffle(seq)
    seq1=[pair[0] for pair in seq]
    seq2=[pair[1] for pair in seq]
    seq3=[pair[2] for pair in seq]
    avg = len(seq1) / float(num_batches)
    out1 = []
    last = 0.0

    while last < len(seq1):
        out1.append(seq1[int(last):int(last + avg)])
        last += avg
    avg = len(seq2) / float(num_batches)
    out2 = []
    last = 0.0

    while last < len(seq2):
        out2.append(seq2[int(last):int(last + avg)])
        last += avg
    avg = len(seq3) / float(num_batches)
    out3 = []
    last = 0.0

    while last < len(seq3):
        out3.append(seq3[int(last):int(last + avg)])
        last += avg
    return [np.asarray(batch) for batch in out1],[np.asarray(batch) for batch in out2],[np.asarray(batch) for batch in out3]


    

epochs=30
num_batches=30    # note that it's num_batches and not batch_size
batch_word1,batch_word2,batch_y=generate_batches(word_pairs,num_batches)

word1,word2,target,output,loss=architecture(allowed_chars,sequence_length)
optimizer = tf.train.AdamOptimizer().minimize(loss)
print("model loaded\n\n")
losses=[]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver=tf.train.Saver()
    for epoch in range(epochs):
        print("Epoch: "+str(epoch),end=" ")
        for batch in range(len(batch_y)):
            _, l = sess.run([optimizer, loss], feed_dict = {word1: batch_word1[batch], word2: batch_word2[batch], target: batch_y[batch]})
            losses.append(l)
        print("loss: ",l)
    saver.save(sess,"/saved/model.ckpt")
    print("session saved\n")
    while input("continue(y/n) ").lower()!="n":
        w1=input("word1:")
        w2=input("word2:")
        w1=np.asarray([word2mat(w1,allowed_chars,sequence_length)])
        w2=np.asarray([word2mat(w2,allowed_chars,sequence_length)])
        print(output.eval({word1:w1,word2:w2}))
        