import pandas as pd
import tensorflow as tf
import numpy as np
from dataTransform import get_lstm_data

stockCodeList = ['002371', '600036', '600048']
rootPath = '.././'
targets = ['Class_1', 'Class_2', 'Class_4', 'Class_8']
n_steps = 12    # duration

for stockCode in stockCodeList:
    inputFile = rootPath + stockCode + 'train.csv'
    data = pd.read_csv(inputFile)
    
    predictors = []
    for col in data.columns:
        if (col not in targets) & (col != 'DateTime'):
            predictors.append(col)

    for target in targets:
        X_class = []
        y_class = []
        for i in np.eye(4):
            feaData, tarData = get_lstm_data(data, predictors, target, n_steps, 'classification')
            l = np.where((tarData == i).all(1))[0]
            X_class.append(feaData[l])
            y_class.append(tarData[l])
        
        tf.reset_default_graph()
        savingPath = '{0}save/{1}/{2}'.format(rootPath, stockCode, target)
        saver = tf.train.import_meta_graph(savingPath+'/LSTM_model.meta')
        with tf.Session() as sess:
            saver.restore(sess, savingPath+'/LSTM_model')
            print('Accuracy Report ' + stockCode)
            X, y, batch_size = tf.get_collection('input')
            accuracy = tf.get_collection('accuracy')[0]
            for j in range(4):
                p = data[data[target] == j].shape[0] / data.shape[0]
                score = sess.run(accuracy, feed_dict={X: X_class[j], y: y_class[j], batch_size: X_class[j].shape[0]})
                print(target + '=' + str(j), 'proportion=' + str(p), 'accuracy=' + str(score))
