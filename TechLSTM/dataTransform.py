import numpy as np
from sklearn.preprocessing import scale, OneHotEncoder
import pandas as pd


def get_lstm_data(dat, fea, label, step, method):
    
    fea_data = dat[fea].values
    tar_data = dat[label].values
    
    fea_data = scale(fea_data)
    
    if len(tar_data.shape) == 1:
        tar_data = tar_data.reshape(-1, 1)
    
    fea_trans = []
    for row in range(fea_data.shape[0]-step):
        fea_trans.append(fea_data[row:row+step])
    fea_trans = np.array(fea_trans)
    if len(fea_trans.shape) == 2:
        fea_trans = fea_trans.reshape(-1, step, 1)
    
    if method == 'classification':
        ohe = OneHotEncoder()
        ohe.fit(np.unique(tar_data, axis=0))
        tar_trans = ohe.transform(tar_data).toarray()
    else:
        tar_trans = tar_data

    tar_trans = tar_trans[step:]
    return fea_trans, tar_trans


if __name__ == '__main__':

    # filePath = '.././**.csv'
    # data = pd.read_csv(filePath)
    # labels = ['Class_1', 'Class_2', 'Class_4', 'Class_8']
    # target = 'Class_1'
    # features = []
    # for col in data.columns:
        # if (col not in labels) & (col != 'DateTime'):
            # features.append(col)
    # x, y = get_lstm_data(data, features, labels, 20, 'classification')
    # x, y = get_lstm_data(data, features, target, 20, 'classification')
    # print(x.shape)
    # print(y.shape)
    
    filePath = '.././**.csv'
    data = pd.read_csv(filePath, engine='python', skipfooter=1)
    target = 'close'
    feature = target
    x, y = get_lstm_data(data, feature, target, 20, 'regression')
    print(x.shape)
    print(y.shape)
