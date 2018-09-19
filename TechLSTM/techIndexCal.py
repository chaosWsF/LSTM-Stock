import pandas as pd
import numpy as np
import talib


def get_label(revenue, period_length):

    if period_length == 1:
        if revenue <= 0.1:
            lab = 0
        elif 0.1 < revenue <= 1:
            lab = 1
        elif 1 < revenue <= 3:
            lab = 2
        else:
            lab = 3
    elif period_length == 2:
        if revenue <= 0.1:
            lab = 0
        elif 0.1 < revenue <= 2:
            lab = 1
        elif 2 < revenue <= 5:
            lab = 2
        else:
            lab = 3
    elif period_length == 4:
        if revenue <= 0.2:
            lab = 0
        elif 0.2 < revenue <= 3:
            lab = 1
        elif 3 < revenue <= 8:
            lab = 2
        else:
            lab = 3
    else:
        if revenue <= 0.3:
            lab = 0
        elif 0.3 < revenue <= 4:
            lab = 1
        elif 4 < revenue <= 12:
            lab = 2
        else:
            lab = 3

    return lab


def get_info(dataset, info_name):
    info_array = np.array([float(x) for x in dataset[info_name]])
    return info_array
 
 
# stockCode = '002371'
# stockCode = '600036'
stockCode = '600048'

kMultiples = [0.5, 1, 3, 6, 12, 24, 48, 96]
dayPeriods = [1, 2, 4, 8]


startDate = '2017-08-01'
endDate = '2018-08-01'

rootPath = '.././'
inputFile = rootPath + stockCode + '**.csv'
data = pd.read_csv(inputFile, engine='python', skipfooter=1)

data['DateTime'] = pd.to_datetime(data['DateTime'])
close = get_info(data, 'close')
high = get_info(data, 'high')
low = get_info(data, 'low')
volume = get_info(data, 'volume')

df = pd.DataFrame(data['DateTime'].values, columns=['DateTime'])

for k in kMultiples:
    macd1, macd2, _ = talib.MACD(close, fastperiod=10*k, slowperiod=22*k, signalperiod=8*k)
    macd3, macd4, _ = talib.MACD(close, fastperiod=12*k, slowperiod=26*k, signalperiod=9*k)
    macdDiff1 = macd1 - macd2
    macdDiff2 = macd3 - macd4
    feaName1 = 'MACD1_' + str(k)
    feaName2 = 'MACD2_' + str(k)
    feaName3 = 'ADOSC_' + str(k)
    df[feaName1] = macdDiff1
    df[feaName2] = macdDiff2
    if k >=1:
        df[feaName3] = talib.ADOSC(high, low, close, volume, fastperiod=3*k, slowperiod=10*k)

for dayPeriod in dayPeriods:
    dayPeriod_min = int(dayPeriod * 4 * 60 / 5)
    labelName = 'Class_' + str(dayPeriod)
    labelList = [np.nan] * dayPeriod_min
    for i in range(dayPeriod_min, len(close)):
        j = i - dayPeriod_min
        rev = (close[i] - close[j]) / close[j] * 100
        label = get_label(rev, dayPeriod)
        labelList.append(label)
    df[labelName] = np.array(labelList)

mask = (df['DateTime'] > startDate) & (df['DateTime'] < endDate)
df = df.loc[mask]

writingPath = rootPath + stockCode + 'train.csv'
df.to_csv(writingPath, index=False)
