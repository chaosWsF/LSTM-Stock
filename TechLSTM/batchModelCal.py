import pandas as pd
import tensorflow as tf
import numpy as np
from dataTransform import get_lstm_data
# from graphConstruct import lstm_graph
# import glob
import os
import json


def reset_graph(seed):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
stockCode = '002371'
# stockCode = '600036'
# stockCode = '600048'

target = 'Class_1'
# target = 'Class_2'
# target = 'Class_4'
# target = 'Class_8'

rootPath = '.././'
inputFile = rootPath + stockCode + 'train.csv'
data = pd.read_csv(inputFile)
targets = ['Class_1', 'Class_2', 'Class_4', 'Class_8']
predictors = []
for col in data.columns:
    if (col not in targets) & (col != 'DateTime'):
        predictors.append(col)

n_inputs = len(predictors)
n_outputs = len(np.unique(data[target].values))
rs = 803    # random seed

for paramFile in os.listdir('./parameters'):
    with open('./parameters/' + paramFile, 'r') as file:
        params = json.load(file)

    n_steps = params['n_steps']
    n_neurons = params['n_neurons']
    n_layers = params['n_layers']
    lr = params['lr']
    # n_epochs = params['n_epochs']
    bs = params['bs']
    train_keep_prob = params['training_keep_prob']

    feaData, tarData = get_lstm_data(data, predictors, target, n_steps, 'classification')
    n_iterations = (feaData.shape[0] // bs) - 1
    trainSize = n_iterations * bs
    X_train = feaData[:trainSize]
    X_test = feaData[trainSize:]
    y_train = tarData[:trainSize]
    y_test = tarData[trainSize:]

    reset_graph(rs)

    with tf.name_scope('inputs'):
        X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='X_input')
        y = tf.placeholder(tf.float32, [None, n_outputs], name='y_input')
        keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob_input')
        batch_size = tf.placeholder(tf.int32, shape=(), name='batch_size_input')
        tf.add_to_collection('input', X)
        tf.add_to_collection('input', y)
        tf.add_to_collection('input', batch_size)

    with tf.name_scope('weights'):
        w = tf.Variable(tf.truncated_normal([n_neurons, n_outputs], stddev=0.1, seed=rs), dtype=tf.float32, name='W')
        tf.summary.histogram('output_layer_weights', w)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.random_normal([n_outputs], seed=rs), name='b')
        tf.summary.histogram('output_layer_biases', b)

    cells = [tf.contrib.rnn.BasicLSTMCell(n_neurons) for _ in range(n_layers)]
    with tf.name_scope('lstm_dropout'):
        cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob, seed=rs) for cell in cells]
    with tf.name_scope('lstm_cells_layers'):
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells_drop, state_is_tuple=True)
    init_state = multi_layer_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, initial_state=init_state, dtype=tf.float32)

    with tf.name_scope('output_layer'):
        pred = tf.nn.softmax(tf.matmul(outputs[:, -1, :], w) + b)
        # tf.add_to_collection('prediction', pred)
        tf.summary.histogram('outputs', pred)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), axis=[1]))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        training_op = optimizer.minimize(loss)

    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.add_to_collection('accuracy', accuracy)
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    # X, y, keep_prob, batch_size, training_op, merged = lstm_graph(n_steps, n_inputs, n_outputs, n_neurons, n_layers, lr, rs)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver()

    savingPath = '{0}save/{1}/{2}/{3}'.format(rootPath, stockCode, target, paramFile[:-5])

    with tf.Session() as sess:
        sess.run(init)
        trainWriter = tf.summary.FileWriter(savingPath+'/logs/train', sess.graph)
        testWriter = tf.summary.FileWriter(savingPath+'/logs/test', sess.graph)
        # for epoch in range(n_epochs):
        for iteration in range(n_iterations):
            i1 = bs * iteration
            i2 = i1 + bs
            X_batch = X_train[i1:i2]
            y_batch = y_train[i1:i2]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, keep_prob: train_keep_prob, batch_size: bs})
            if iteration % 5 == 0:
                resTrain = sess.run(merged, feed_dict={X: X_batch, y: y_batch, batch_size: bs})
                resTest = sess.run(merged, feed_dict={X: X_test, y: y_test, batch_size: X_test.shape[0]})
                trainWriter.add_summary(resTrain, iteration)
                testWriter.add_summary(resTest, iteration)
        saver.save(sess, savingPath+'/LSTM_model')
        print(paramFile, 'Done')
