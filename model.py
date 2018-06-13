# -*- coding: utf-8 -*-
"""
Created on Fri May  4 09:58:45 2018

@author: soumil
"""

import tensorflow as tf

def LSTM(x,scope,n_hidden):
    n_layers=3
    x=tf.unstack(tf.transpose(x,perm=[1,0,2]))
    with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
        stacked_rnn_fw=[]
        for _ in range(n_layers):
            fw_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
            stacked_rnn_fw.append(fw_cell)
        lstm_fw_cell_m=tf.nn.rnn_cell.MultiRNNCell(stacked_rnn_fw)
    with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
        stacked_rnn_bw=[]
        for _ in range(n_layers):
            bw_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
            stacked_rnn_bw.append(bw_cell)
        lstm_bw_cell_m=tf.nn.rnn_cell.MultiRNNCell(stacked_rnn_bw)
        outputs,_,_=tf.nn.static_bidirectional_rnn(lstm_fw_cell_m,lstm_bw_cell_m,x,dtype=tf.float32)
    return outputs[-1]
def architecture(allowed_chars=list("qwertyuiopasdfghjklzxcvbnm1234567890-_' "),sequence_length=30,hidden_units=32):

    num_allowed_chars=len(allowed_chars)
    
    tf.reset_default_graph()
    word1=tf.placeholder(tf.float32,[None,num_allowed_chars,sequence_length],name='input_x1') #shape (batch_size,num_allowed_chars,sequence_length)
    word2=tf.placeholder(tf.float32,[None,num_allowed_chars,sequence_length],name='input_x2') #shape (batch_size,num_allowed_chars,sequence_length)
    target=tf.placeholder(tf.float32,[None,1],name='input_y') #shape (batch_size,1)
    
    with tf.name_scope("lstm_output"):
        lstm_out1=LSTM(word1,"side1",hidden_units)
        lstm_out2=LSTM(word2,"side2",hidden_units)
        lstm_out=tf.concat([lstm_out1,lstm_out2],axis=1,name="lstm_out") #shape (batch_size,4*n_hidden)
        
        w1=tf.Variable(tf.random_normal([4*hidden_units,32],name='w1'))
        w2=tf.Variable(tf.random_normal([32,1],name='w2'))
        b1=tf.Variable(tf.random_normal([32],name='b1'))
        b2=tf.Variable(tf.random_normal([1],name='b2'))
        
        hidden1=tf.nn.relu(tf.add(tf.matmul(lstm_out,w1),b1))
        output=tf.add(tf.matmul(hidden1,w2),b2)
        loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output,labels=target))
        
    return word1,word2,target,output,loss
