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
        
        distance=tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(lstm_out1,lstm_out2)),1,keep_dims=True))
        output=1-tf.nn.tanh(distance)
        loss=tf.losses.mean_squared_error(target,output)
    return word1,word2,target,output,loss
