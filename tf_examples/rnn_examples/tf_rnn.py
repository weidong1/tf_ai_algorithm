#!/usr/bin/env python
# encoding: utf-8
'''
@file: tf_rnn.py
@desc: this is an example for classification using LSTM
         RNN中的cell可以选择是普通的rnn/LSTM或是GRU
'''


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(1)   # set random seed

# 导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001                  # learning rate
training_iters = 100000     # train step 上限
batch_size = 128
n_inputs = 28               # 数据维度，图像上可以理解为列的个数
n_steps = 28                # time steps, 步长，可以理解为文本中的window length
n_hidden_units = 128        # neurons in unit of hidden layer
n_classes = 10              # MNIST classes (0-9 digits)

# x y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs]) # (batch_size, time_steps, inputs_size)
y = tf.placeholder(tf.float32, [None, n_classes])

# 对 weights biases 初始值的定义
weights = {
    # shape (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):
    # X是一个batch的Image, 维度是 batch_num * image_height * image_width
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batches * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # X_in = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units]) # 这个input格式是 [batch_size, time_steps, num_hidden_units]

    # 使用 basic LSTM Cell.
    # num_units: int类型，LSTM单元中的神经元数量，即输出神经元数量
    # forget_bias: float类型，偏置增加了忘记门。从CudnnLSTM训练的检查点(checkpoin) 恢复时，必须手动设置
    # state_is_tuple: 如果为True，则接受和返回的状态是c_state和m_state的2 - tuple；如果为False，则他们沿着列轴连接。后一种即将被弃用。
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)  # 初始化全零 state， zero_state（batch_size，dtype）两个参数
    # tf.nn.dynamic_rnn(cell,inputs,sequence_length=None,initial_state=None,dtype=None,parallel_iterations=None,swap_memory=False,time_major=False,scope=None)
    # 这是tensoflow针对RNN的LSTM提供的函数
    # input format格式为[batch_size, max_time, embed_size]，
    #       其中batch_size是输入的这批数据的数量，max_time就是这批数据中序列的最长长度，embed_size表示嵌入的词向量的维度。
    # time_major 如果是True，output的维度是[steps, batch_size, embed_size]，反之就是[batch_size, max_time, embed_size]。就是和输入是一样的
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False) # 一般用dynamic而不是static
    # 当只有一层RNN时, output 包含了隐含层所有时刻(即所有steps)的输出，如果加层的话，那么这个output 的每个时刻，就作为下一层每个时刻的输入；
    # 一般情况下states的形状为[batch_size, cell.output_size]，
    # 但当输入的cell为BasicLSTMCell时，state的形状为[2，batch_size, cell.output_size]，其中2也对应着LSTM中的cell state和hidden state。
    # final_state就是整个LSTM输出的最终的状态（即最后一个时刻的输出），包含c和h。c和h的维度都是[batch_size， n_hidden]，和output的最后一个时刻值一样；
    # 因为要取outputs中最后一个时刻的值，所以把 outputs转置一下，再拼接在一起 变成 列表 [(batch, outputs)..] * steps
    outputs2 = tf.unstack(tf.transpose(outputs, [1, 0, 2]))  #dynamic_rnn取outputs一定要用unstack把多个time_step的结果分开再取最后一个time的结果
                                                            #  final_state包括两个向量分别为LSTM_O，LSTM_S，大小都为[1，128]。
                                                            # 其中LSTM_O为RNN模型的输出，LSTM_S为RNN模型的内部记忆向量，传递到下一个RNN神经元。
                                                            # 最后对LSTM_O进行Softmax处理，通过概率分析出该样本的类别。
    results = tf.matmul(outputs2[-1], weights['out']) + biases['out']  # 选取最后一个 output
    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
        }))
        step += 1


