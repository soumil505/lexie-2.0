# -*- coding: utf-8 -*-
"""
Created on Mon May 28 22:21:48 2018

@author: soumil
"""

import tensorflow as tf
from model import architecture
from preprocessing import word2mat
import random
import numpy as np
import requests
from bs4 import BeautifulSoup



def get_synonyms_antonyms(word):
    url = "http://www.thesaurus.com/browse/" + word +"?s=t"
    r = requests.get(url)
    soup = BeautifulSoup(r.content,"lxml")
    word_data = soup.find_all("a",{"class":"css-1hn7aky e1s2bo4t1"})
    syns = [w.text for w in word_data]
    word_data = soup.find_all("a",{"class":"css-1usnxsl e1s2bo4t1"})
    ants = [w.text for w in word_data]
    return syns,ants


allowed_chars="qwertyuiopasdfghjklzxcvbnm'-_1234567890 "

sequence_length=30

def generate_word_pairs(words):
    pairs=[]
    for word in words:
        syn,ant=get_synonyms_antonyms(word)
        pairs+=[(word,synonym,[1]) for synonym in syn]
        pairs+=[(word,antonym,[0]) for antonym in ant]
        typo=list(word)
        typo[random.randint(0,len(typo)-1)]=random.choice(allowed_chars)
        pairs+=[(word,''.join(typo),[1])]
    return pairs

def generate_word_matrix_pairs(word_pairs,allowed_chars,sequence_length=30):
    mat_pairs=[]
    for word_pair in word_pairs:
        if len(word_pair[0])>sequence_length or len(word_pair[1])>sequence_length:
            continue
        mat_pairs.append([word2mat(word_pair[0],allowed_chars,sequence_length),word2mat(word_pair[1],allowed_chars,sequence_length),word_pair[2]])
    return mat_pairs
def generate_batches(seq,num_batches=30,seed=0):
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

def get_word():
    word=""
    while len(word.split())!=1:
        url="http://www.thesaurus.com/list/"+random.choice("qwertyuiopasdfghjklzxcvbnm")+"/"+random.choice("123456789")
        r=requests.get(url)
        soup=BeautifulSoup(r.content,"lxml")
        word_data = soup.find_all("span",{"class":"word"})
        word=random.choice(word_data).text
    return word
        
epochs=100
num_words=30
num_batches=10
hidden_units=16


word1,word2,target,output,loss=architecture(allowed_chars,sequence_length,hidden_units)
optimizer = tf.train.AdamOptimizer().minimize(loss)
saver=tf.train.Saver()
print("model loaded\n\n")


with tf.Session() as sess:
    while True:
        words=[get_word() for _ in range(num_words)]
        print(words)
        word_pairs=generate_word_pairs(words)
        seq=generate_word_matrix_pairs(word_pairs,allowed_chars,sequence_length)
        batch_word1,batch_word2,batch_y=generate_batches(seq,num_batches)
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            print("Epoch: "+str(epoch),end=" ")
            for batch in range(len(batch_y)):
                _, l = sess.run([optimizer, loss],
                                feed_dict = {word1: batch_word1[batch],
                                             word2: batch_word2[batch],
                                             target: batch_y[batch]})
                
            print("loss: ",l)
        saver.save(sess,"/saved/pretrained.ckpt")
        print("session saved\n")
   
        
