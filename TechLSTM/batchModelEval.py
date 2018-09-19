import pandas as pd
import tensorflow as tf
import numpy as np
from dataTransform import get_lstm_data
import glob
import os

stockCode = '002371'
target = 'Class_1'

rootPath = '.././'
targets = ['Class_1', 'Class_2', 'Class_4', 'Class_8']
n_classValue = 4

inputFile = rootPath + stockCode + 'train.csv'
data = pd.read_csv(inputFile)

predictors = []
for col in data.columns:
    if (col not in targets) & (col != 'DateTime'):
        predictors.append(col)

pp = []
n = data.shape[0]
for j in range(n_classValue):
    pp.append(data[data[target] == j].shape[0] / n)

savingPath = '{0}save/{1}/{2}/p*_*'.format(rootPath, stockCode, target)
modelList = glob.glob(savingPath)
for modelPath in modelList:
    mp = os.path.basename(modelPath)
    n_steps = int(mp[1:].split('_')[0])
    X_class = []
    y_class = []
    for i in np.eye(n_classValue):
        feaData, tarData = get_lstm_data(data, predictors, target, n_steps, 'classification')
        l = np.where((tarData == i).all(1))[0]
        X_class.append(feaData[l])
        y_class.append(tarData[l])
    
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(modelPath+'/LSTM_model.meta')
    with tf.Session() as sess:
        saver.restore(sess, modelPath+'/LSTM_model')
        print('Accuracy Report')
        print(stockCode, target, mp)
        X, y, batch_size = tf.get_collection('input')
        accuracy = tf.get_collection('accuracy')[0]
        for j in range(n_classValue):
            score = sess.run(accuracy, feed_dict={X: X_class[j], y: y_class[j], batch_size: X_class[j].shape[0]})
            print('class=' + str(j), 'proportion=' + str(pp[j]), 'accuracy=' + str(score))
