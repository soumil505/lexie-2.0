# -*- coding: utf-8 -*-
"""
Created on Sat May 26 09:58:48 2018

@author: soumil
"""
import numpy as np

def remove_stopwords(text,stopwords):
    text=text.lower().split()
    return ' '.join(list(set([word for word in text if word not in stopwords])))

def remove_disallowed_chars(text,allowed_chars):
    text=text.lower()
    if " " not in allowed_chars:
        raise Exception("Character ' ' (space) should be in allowed_chars")
    return ''.join([char for char in text if char in allowed_chars])

def char_dict(allowed_chars):
    if " " not in allowed_chars:
        raise Exception("Character ' ' (space) should be in allowed_chars")
    allowed_chars=list(allowed_chars)
    num_allowed_chars=len(allowed_chars)
    vecs=[]
    for i in range(num_allowed_chars):
        zero=[0 for j in range(num_allowed_chars)]
        zero[i]=1
        vecs.append(np.asarray(zero))
    char2vec={char:vec for char,vec in zip(allowed_chars,vecs)}
    return char2vec

def word2mat(word,allowed_chars,sequence_length=30):
    word=remove_disallowed_chars(word,allowed_chars)
    char2vec=char_dict(allowed_chars)
    mat=np.asarray([char2vec[char] for char in list(word)])
    mat=np.transpose(mat)
    zero=np.zeros((np.shape(mat)[0],sequence_length))
    zero[:mat.shape[0],:mat.shape[1]] = mat
    return zero

def generate_word_pairs(text,allowed_chars,stopwords=[],grouped_words=1):
    text=remove_disallowed_chars(text,allowed_chars)
    text=remove_stopwords(text,stopwords)
    text=text.split()
    grouped_text=[]
    for i in range(len(text)-grouped_words+1):
        word=text[i]
        for j in range(0,grouped_words-1):    
            word=word+" "+text[i+j+1]
        grouped_text.append(word)
    word_pairs=[]
    if len(grouped_text)>=13:
        for i in range(6,len(grouped_text)-6):
            word_pairs.append([grouped_text[i],grouped_text[i-6],[0.4]])
            word_pairs.append([grouped_text[i],grouped_text[i-5],[0.5]])
            word_pairs.append([grouped_text[i],grouped_text[i-4],[0.6]])
            word_pairs.append([grouped_text[i],grouped_text[i-3],[0.7]])
            word_pairs.append([grouped_text[i],grouped_text[i-2],[0.8]])
            word_pairs.append([grouped_text[i],grouped_text[i-1],[0.9]])
            word_pairs.append([grouped_text[i],grouped_text[i],[1]])
            word_pairs.append([grouped_text[i],grouped_text[i+1],[0.9]])
            word_pairs.append([grouped_text[i],grouped_text[i+2],[0.8]])
            word_pairs.append([grouped_text[i],grouped_text[i+3],[0.7]])
            word_pairs.append([grouped_text[i],grouped_text[i+4],[0.6]])
            word_pairs.append([grouped_text[i],grouped_text[i+5],[0.5]])
            word_pairs.append([grouped_text[i],grouped_text[i+6],[0.4]])
    return word_pairs

def generate_word_matrix_pairs(text,allowed_chars,stopwords=[],grouped_words=1,sequence_length=30):
    word_pairs=generate_word_pairs(text,allowed_chars,stopwords,grouped_words)
    mat_pairs=[]
    for word_pair in word_pairs:
        if len(word_pair[0])>sequence_length or len(word_pair[1])>sequence_length:
            continue
        mat_pairs.append([word2mat(word_pair[0],allowed_chars,sequence_length),word2mat(word_pair[1],allowed_chars,sequence_length),word_pair[2]])
    return mat_pairs
    