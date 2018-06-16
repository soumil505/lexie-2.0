# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:31:01 2018

@author: soumil
"""

import tensorflow as tf
from model import architecture
from preprocessing import word2mat
import numpy as np


allowed_chars="qwertyuiopasdfghjklzxcvbnm'-_1234567890 "

sequence_length=30

hidden_units=16




word1,word2,target,output,loss=architecture(allowed_chars,sequence_length,hidden_units)
optimizer = tf.train.AdamOptimizer().minimize(loss)
saver=tf.train.Saver()
print("model loaded\n\n")


with tf.Session() as sess:
    saver.restore(sess,"/saved/pretrained.ckpt")
    print("session restored")
    while input("continue(y/n) ").lower()!="n":
        w1=input("word1:")
        w2=input("word2:")
        w1=np.asarray([word2mat(w1,allowed_chars,sequence_length)])
        w2=np.asarray([word2mat(w2,allowed_chars,sequence_length)])
        print(tf.nn.sigmoid(output).eval({word1:w1,word2:w2}))
