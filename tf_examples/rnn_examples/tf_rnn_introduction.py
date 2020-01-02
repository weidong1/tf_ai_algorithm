#!/usr/bin/env python
# encoding: utf-8
'''
@file: tf_rnn_introduction.py
@desc: 介绍tensorflow中RNN的几个basic类的用法
'''

import tensorflow as tf
import numpy as np

# 1 RNNCell
# RNNCell是一个抽象类，有两个子类BasicRNNCell和BasicLSTMCell
# 1.1 BasicRNNCell
basic_rnncell = tf.nn.rnn_cell.BasicRNNCell(num_units=128) # state_size = 128，每个cell输出的也是一个向量
print(basic_rnncell.state_size) # 128
inputs = tf.placeholder(np.float32, shape=(32, 100)) # 输入数据形状是(batch_size, input_size)
h0 = basic_rnncell.zero_state(32, np.float32) # 得到一个全0的初始状态，形状为(batch_size, state_size)
output, h1 = basic_rnncell(inputs, h0) #隐式用call函数
print(h1.shape) # (32, 128) 输出是(batch_size, output_size)，隐层状态是(batch_size, state_size)
# 1.2 BasicLSTMCell
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
batch_size = 32
inputs = tf.placeholder(np.float32, shape=(batch_size, 100)) # 32 是 batch_size
h0 = lstm_cell.zero_state(batch_size, np.float32) # 通过zero_state得到一个全0的初始状态
output, h1 = lstm_cell(inputs, h0) # lstm有两个隐状态h和c,形状都是(batch_size, state_size)
print(h1.h)  # shape=(32, 128)
print(h1.c)  # shape=(32, 128)
# 1.3 dynamic_rnn
print('--- dynamic rnn')
# dynamic_rnn不用对每个hidden unit单独调，可以一次性调用得到所有的隐状态和输出
# inputs: shape = (batch_size, time_steps, input_size)，其中time_steps表示序列本身的长度
# cell: RNNCell或是其他的Cell
# initial_state: shape = (batch_size, cell.state_size)。初始状态。一般可以取零矩阵
#outputs就是time_steps步里所有的输出。它的形状为(batch_size, time_steps, cell.output_size)。
#state是最后一个cell输出的隐状态，形状会随着cell的类型而变化，通常为[batch_size, cell.output_size ]，也可能是tuple分别是h和c
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
inputs = tf.placeholder(np.float32, shape=(batch_size,10, 100))
init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
outputs, state = tf.nn.dynamic_rnn(lstm_cell, inputs, initial_state=init_state)
print(outputs)
print(state)








