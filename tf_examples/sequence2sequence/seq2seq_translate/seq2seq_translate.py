#!/usr/bin/env python
# encoding: utf-8
'''
@file: seq2seq_translate.py
@desc:  translate english to french
        refer to https://zhuanlan.zhihu.com/p/37148308

'''


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm


# -- 2 - Data Process
# --------2.1 - 加载原始数据与目标数据
# English source data
with open("data/small_vocab_en", "r", encoding="utf-8") as f:
    source_text = f.read()

# French target data
with open("data/small_vocab_fr", "r", encoding="utf-8") as f:
    target_text = f.read()

# ------ 2.3 - 数据预处理
''' 先对语料进行处理。即将文本语料转化为机器所能识别的数字。
例如，对英文句子：I love machine learning and deep learning.编码为数字[28, 29, 274, 873, 12, 983, 873]。
step1 根据语料构造英文与法语的字典（vocabulary）
step2 构造英语与法语的映射，即将单词转换为数字的字典
step3 构造英语与法语的反向映射，即从数字转换为单词的字典
'''
# 构造英文词典
source_vocab = list(set(source_text.lower().split()))
# 构造法文词典
target_vocab = list(set(target_text.lower().split()))

# 特殊字符
SOURCE_CODES = ['<PAD>', '<UNK>']
TARGET_CODES = ['<PAD>', '<EOS>', '<UNK>', '<GO>']  # 在target中，需要增加<GO>与<EOS>特殊字符

# 构造英文映射字典
source_vocab_to_int = {word: idx for idx, word in enumerate(SOURCE_CODES + source_vocab)}
source_int_to_vocab = {idx: word for idx, word in enumerate(SOURCE_CODES + source_vocab)}

# 构造法语映射词典
target_vocab_to_int = {word: idx for idx, word in enumerate(TARGET_CODES + target_vocab)}
target_int_to_vocab = {idx: word for idx, word in enumerate(TARGET_CODES + target_vocab)}

# -------  2.4 - 语料转换
# 将文本语料转成数字id, 规则见text_to_int函数
def text_to_int(sentence, map_dict, max_length=20, is_target=False):
    """
    对文本句子进行数字编码

    @param sentence: 一个完整的句子，str类型
    @param map_dict: 单词到数字的映射，dict
    @param max_length: 句子的最大长度
    @param is_target: 是否为目标语句。在这里要区分目标句子与源句子，因为对于目标句子（即翻译后的句子）我们需要在句子最后增加<EOS>
    """

    # 用<PAD>填充整个序列
    text_to_idx = []
    # unk index
    unk_idx = map_dict.get("<UNK>")
    pad_idx = map_dict.get("<PAD>")
    eos_idx = map_dict.get("<EOS>")

    # 如果是输入源文本
    if not is_target:
        for word in sentence.lower().split():
            text_to_idx.append(map_dict.get(word, unk_idx))

    # 否则，对于输出目标文本需要做<EOS>的填充最后
    else:
        for word in sentence.lower().split():
            text_to_idx.append(map_dict.get(word, unk_idx))
        text_to_idx.append(eos_idx)

    # 如果超长需要截断
    if len(text_to_idx) > max_length:
        return text_to_idx[:max_length]
    # 如果不够则增加<PAD>
    else:
        text_to_idx = text_to_idx + [pad_idx] * (max_length - len(text_to_idx))
        return text_to_idx

# 对源句子进行转换 Tx = 20
source_text_to_int = []
for sentence in tqdm.tqdm(source_text.split("\n")):
    source_text_to_int.append(text_to_int(sentence, source_vocab_to_int, 20,
                                          is_target=False))
# 对目标句子进行转换  Ty = 25
target_text_to_int = []
for sentence in tqdm.tqdm(target_text.split("\n")):
    target_text_to_int.append(text_to_int(sentence, target_vocab_to_int, 25,
                                          is_target=True))

X = np.array(source_text_to_int)
Y = np.array(target_text_to_int)


# -- 3 构建模型
''' 模型输入 model_inputs
    Encoder端 encoder_layer
    Decoder端
    Decoder输入端 decoder_layer_inputs
    Decoder训练 decoder_layer_train
    Decoder预测/推断 decoder_layer_inference
    Seq2Seq模型
'''
# ------ 3.1 - 模型输入
def model_inputs():
    """
    构造输入tensor
    inputs是一个二维tensor，第一个维度代表batch size，第二个维度是序列长度，targets同理
    返回：inputs, targets, learning_rate, source_sequence_len, target_sequence_len, max_target_sequence_len，类型为tensor
    """
    inputs = tf.placeholder(tf.int32, [None, None], name="inputs")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    source_sequence_len = tf.placeholder(tf.int32, (None,), name="source_sequence_len")
    target_sequence_len = tf.placeholder(tf.int32, (None,), name="target_sequence_len")
    max_target_sequence_len = tf.placeholder(tf.int32, (None,), name="max_target_sequence_len")

    return inputs, targets, learning_rate, source_sequence_len, target_sequence_len, max_target_sequence_len

# ------- 3.2 - Encoder端
def encoder_layer(rnn_inputs, rnn_size, rnn_num_layers,
                  source_sequence_len, source_vocab_size, encoder_embedding_size=100):
    """
    构造Encoder端
    @param rnn_inputs: rnn的输入，形状为[batch_size, doc_length]的int32或int64张量，也就是经过预处理的输入数据
    @param rnn_size: rnn的隐层结点数
    @param rnn_num_layers: rnn的堆叠层数
    @param source_sequence_len: 英文句子序列的长度
    @param source_vocab_size: 英文词典的大小， Integer number of symbols in vocabulary.
    @param encoder_embedding_size: Encoder层中对单词进行词向量嵌入后的维度
    """
    # 对输入的单词进行词向量嵌入，这是固定做法
    encoder_embed = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoder_embedding_size)

    # LSTM单元
    def get_lstm(rnn_size):
        lstm = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
        return lstm

    # 堆叠rnn_num_layers层LSTM
    lstms = tf.contrib.rnn.MultiRNNCell([get_lstm(rnn_size) for _ in range(rnn_num_layers)])
    encoder_outputs, encoder_states = tf.nn.dynamic_rnn(lstms, encoder_embed, source_sequence_len,
                                                        dtype=tf.float32)

    return encoder_outputs, encoder_states

# -- 3.2 - Decoder端
'''在Decoder端，我们分为三个模块：
    decoder_layer_inputs：主要用来处理Decoder的输入；
    decoder_layer_train：用于训练过程中的Decoder；
    decoder_layer_infer：用于测试过程中的Decoder。'''
# --------- 3.2.1 - Decoder Layer Inputs
def decoder_layer_inputs(target_data, target_vocab_to_int, batch_size):
    """
    对Decoder端的输入进行处理
    @param target_data: 法语数据的tensor
    @param target_vocab_to_int: 法语数据的词典到索引的映射
    @param batch_size: batch size

    在Decoder端需要用“<GO>”告诉它翻译的开始，
    在Decoder最后一个阶段时，根据上一轮输入结果和hidden state得到输出<EOS>，代表着翻译的结束。
    这最后的输出词并没有再作为输入来进行预测，也就是说<EOS>并不会作为下一阶段的输入，所以要将输出结果的最后一个单词去掉。
    """
    # 去掉batch中每个序列句子的最后一个单词
    ''' 
    tf.strided_slice(input,begin,end,strides=None)提取张量的一个分段切片 
    从给定的 input_ 张量中提取一个尺寸 (end-begin)/stride 的片段.从 begin 片段指定的位置开始,继续添加 stride 索引,直到所有维度都不小于 end.
    '''
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    # 在batch中每个序列句子的前面添加”<GO>"
    decoder_inputs = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int["<GO>"]),
                                ending], 1)
    return decoder_inputs

# ---------- 3.2.2 - Decoder Traing
def decoder_layer_train(encoder_states, decoder_cell, decoder_embed,
                        target_sequence_len, max_target_sequence_len, output_layer):
    """
    Decoder端的训练
    @param encoder_states: Encoder端编码得到的Context Vector
    @param decoder_cell: Decoder端
    @param decoder_embed: Decoder端词向量嵌入后的输入
    @param target_sequence_len: 法语文本的长度
    @param max_target_sequence_len: 法语文本的最大长度
    @param output_layer: 输出层
    """
    # 生成helper对象，并传入BasicDecoder中。
    # training用TrainingHelper， 测试用GreedyEmbeddingHelper，这是固定用法
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed,
                                                        sequence_length=target_sequence_len,
                                                        time_major=False)
    ''' 
    BasicDecoder(cell, helper, initial_state, output_layer=None):
      cell: 一个 `RNNCell` 实例.
      helper: 一个 `Helper` 实例.
      initial_state: RNNCell 的初始状态.，一个 (可能组成一个tulpe)tensors 和 TensorArrays.
             是用Encoder的最后一个隐层状态
      output_layer: (可选) 一个 `tf.layers.Layer` 实例, 例如：`tf.layers.Dense`. 应用于RNN 输出层之前的可选层,用于存储结果或者采样.
    '''
    training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                       training_helper,
                                                       encoder_states,
                                                       output_layer)
    # 在dynamic_decode方法中，impute_finished为True时，当预测被标记为结束时，会自动复制前一轮状态向量，并将输出都置为0。
    # dynamic_decode 函数类似于 dynamic_rnn，帮你自动执行 rnn 的循环，返回完整的输出序列
    # 返回值decoder_outputs是tf.contrib.seq2seq.BasicDecoderOutput类型，包括两个字段：rnn_output，sample_id。
    training_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                       impute_finished=True,
                                                                       maximum_iterations=max_target_sequence_len)
    return training_decoder_outputs

# ----------- 3.2.3 - Decoder Inferenc
def decoder_layer_infer(encoder_states, decoder_cell, decoder_embed, start_id, end_id,
                        max_target_sequence_len, output_layer, batch_size):
    """
    Decoder端的预测/推断
    @param encoder_states: Encoder端编码得到的Context Vector
    @param decoder_cell: Decoder端
    @param decoder_embed: Decoder端词向量嵌入后的输入
    @param start_id: 句子起始单词的token id， 即"<GO>"的编码
    @param end_id: 句子结束的token id，即"<EOS>"的编码
    @param max_target_sequence_len: 法语文本的最大长度
    @param output_layer: 输出层
    @batch_size: batch size
    """
    start_tokens = tf.tile(tf.constant([start_id], dtype=tf.int32), [batch_size], name="start_tokens")
    ''' TensorFlow的接口中对GreedyEmbeddingHelper的定义为：
        A helper for use during inference.
        Uses the argmax of the output (treated as logits) and passes the result through an embedding layer to get the next input.'''
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embed,
                                                                start_tokens,
                                                                end_id)
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                        inference_helper,
                                                        encoder_states,
                                                        output_layer)
    inference_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                        impute_finished=True,
                                                                        maximum_iterations=max_target_sequence_len)
    return inference_decoder_outputs

# ---------- 3.2.4 - Decoder Layer
def decoder_layer(encoder_states, decoder_inputs, target_sequence_len,
                  max_target_sequence_len, rnn_size, rnn_num_layers,
                  target_vocab_to_int, target_vocab_size, decoder_embedding_size, batch_size):
    """
    构造Decoder端
    @param encoder_states: Encoder端编码得到的Context Vector
    @param decoder_inputs: Decoder端的输入
    @param target_sequence_len: 法语文本的长度
    @param max_target_sequence_len: 法语文本的最大长度
    @param rnn_size: rnn隐层结点数
    @param rnn_num_layers: rnn堆叠层数
    @param target_vocab_to_int: 法语单词到token id的映射
    @param target_vocab_size: 法语词典的大小
    @param decoder_embedding_size: Decoder端词向量嵌入的大小
    @param batch_size: batch size
    """
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoder_embedding_size]))
    decoder_embed = tf.nn.embedding_lookup(decoder_embeddings, decoder_inputs)
    def get_lstm(rnn_size):
        lstm = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=456))
        return lstm
    decoder_cell = tf.contrib.rnn.MultiRNNCell([get_lstm(rnn_size) for _ in range(rnn_num_layers)])
    # output_layer logits
    output_layer = tf.layers.Dense(target_vocab_size) # dense层输出的向量维度是整个的vocabulary size
    with tf.variable_scope("decoder"):
        training_logits = decoder_layer_train(encoder_states,
                                              decoder_cell,
                                              decoder_embed,
                                              target_sequence_len,
                                              max_target_sequence_len,
                                              output_layer)
    with tf.variable_scope("decoder", reuse=True):
        inference_logits = decoder_layer_infer(encoder_states,
                                               decoder_cell,
                                               decoder_embeddings,
                                               target_vocab_to_int["<GO>"],
                                               target_vocab_to_int["<EOS>"],
                                               max_target_sequence_len,
                                               output_layer,
                                               batch_size)
    return training_logits, inference_logits

# -- 3.3 - Seq2Seq模型
def seq2seq_model(input_data, target_data, batch_size,
                  source_sequence_len, target_sequence_len, max_target_sentence_len,
                  source_vocab_size, target_vocab_size,
                  encoder_embedding_size, decoder_embeding_size,
                  rnn_size, rnn_num_layers, target_vocab_to_int):
    """
    构造Seq2Seq模型

    @param input_data: tensor of input data
    @param target_data: tensor of target data
    @param batch_size: batch size
    @param source_sequence_len: 英文语料的长度
    @param target_sequence_len: 法语语料的长度
    @param max_target_sentence_len: 法语的最大句子长度
    @param source_vocab_size: 英文词典的大小
    @param target_vocab_size: 法语词典的大小
    @param encoder_embedding_size: Encoder端词嵌入向量大小
    @param decoder_embedding_size: Decoder端词嵌入向量大小
    @param rnn_size: rnn隐层结点数
    @param rnn_num_layers: rnn堆叠层数
    @param target_vocab_to_int: 法语单词到token id的映射
    """
    _, encoder_states = encoder_layer(input_data, rnn_size, rnn_num_layers, source_sequence_len,
                                      source_vocab_size, encoder_embedding_size)

    decoder_inputs = decoder_layer_inputs(target_data, target_vocab_to_int, batch_size)

    training_decoder_outputs, inference_decoder_outputs = decoder_layer(encoder_states,
                                                                        decoder_inputs,
                                                                        target_sequence_len,
                                                                        max_target_sentence_len,
                                                                        rnn_size,
                                                                        rnn_num_layers,
                                                                        target_vocab_to_int,
                                                                        target_vocab_size,
                                                                        decoder_embeding_size,
                                                                        batch_size)
    return training_decoder_outputs, inference_decoder_outputs


# -- 4 Training Model
# ----------- 4.1 - 设置参数
# Number of Epochs
epochs = 10
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 128
# Number of Layers
rnn_num_layers = 1
# Embedding Size
encoder_embedding_size = 100
decoder_embedding_size = 100
# Learning Rate
lr = 0.001
# 每50轮打一次结果
display_step = 50



# -----------  4.2 - 构建图
train_graph = tf.Graph()

with train_graph.as_default():
    inputs, targets, learning_rate, source_sequence_len, target_sequence_len, _ = model_inputs()
    max_target_sequence_len = 25
    train_logits, inference_logits = seq2seq_model(tf.reverse(inputs, [-1]),
                                                   targets,
                                                   batch_size,
                                                   source_sequence_len,
                                                   target_sequence_len,
                                                   max_target_sequence_len,
                                                   len(source_vocab_to_int),
                                                   len(target_vocab_to_int),
                                                   encoder_embedding_size,
                                                   decoder_embedding_size,
                                                   rnn_size,
                                                   rnn_num_layers,
                                                   target_vocab_to_int)

    training_logits = tf.identity(train_logits.rnn_output, name="logits")
    inference_logits = tf.identity(inference_logits.sample_id, name="predictions")

    masks = tf.sequence_mask(target_sequence_len, max_target_sequence_len, dtype=tf.float32, name="masks")

    with tf.name_scope("optimization"):
        '''sequence_loss参数
        logits：尺寸[batch_size, sequence_length, num_decoder_symbols]
        targets：尺寸[batch_size, sequence_length]，不用做one_hot。
        weights：[batch_size, sequence_length]，即mask，滤去padding的loss计算，使loss计算更准确。
         '''
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        gradients = optimizer.compute_gradients(cost)
        clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(clipped_gradients)


# ----- 4.3 - 训练模型
def get_batches(sources, targets, batch_size):
    """
    获取batch
    """
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Need the lengths for the _lengths parameters
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield sources_batch, targets_batch, source_lengths, targets_lengths


with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                get_batches(source_text_to_int, target_text_to_int, batch_size)):

            _, loss = sess.run(
                [train_op, cost],
                {inputs: source_batch,
                 targets: target_batch,
                 learning_rate: lr,
                 source_sequence_len: sources_lengths,
                 target_sequence_len: targets_lengths})

            if batch_i % display_step == 0 and batch_i > 0:
                batch_train_logits = sess.run(
                    inference_logits,
                    {inputs: source_batch,
                     source_sequence_len: sources_lengths,
                     target_sequence_len: targets_lengths})

                print('Epoch {:>3} Batch {:>4}/{} - Loss: {:>6.4f}'
                      .format(epoch_i, batch_i, len(source_text_to_int) // batch_size, loss))

    # Save Model
    saver = tf.train.Saver()
    checkpoint = "./model/trained_model.ckpt"
    saver.save(sess, checkpoint) #
    print('Model Trained and Saved')



# -- 5 预测
def sentence_to_seq(sentence, source_vocab_to_int):
    """
    将句子转化为数字编码
    """
    unk_idx = source_vocab_to_int["<UNK>"]
    word_idx = [source_vocab_to_int.get(word, unk_idx) for word in sentence.lower().split()]

    return word_idx

translate_sentence_text = "i dislike grapefruit , lemons , and peaches ."

translate_sentence = sentence_to_seq(translate_sentence_text, source_vocab_to_int)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_len:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_len:0')

    translate_logits = sess.run(logits, {input_data: [translate_sentence]*batch_size,
                                         target_sequence_length: [len(translate_sentence)*2]*batch_size,
                                         source_sequence_length: [len(translate_sentence)]*batch_size})[0]

print('【Input】')
print('  Word Ids:      {}'.format([i for i in translate_sentence]))
print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

print('\n【Prediction】')
print('  Word Ids:      {}'.format([i for i in translate_logits]))
print('  French Words: {}'.format([target_int_to_vocab[i] for i in translate_logits]))

print("\n【Full Sentence】")
print(" ".join([target_int_to_vocab[i] for i in translate_logits]))






