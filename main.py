import tensorflow as tf 

import pandas as pd
import random
import numpy as np
from util import moving_average


#"import pandas as pd
BASE_DIR = "./data"
FILE_TRAIN_CASES = "./data/train_cases.csv"
df_tc = pd.read_csv(FILE_TRAIN_CASES)
df_tc.head(10)



x_train = None
x_train_cases = None
y_train = None
x_test = None

import os
import pandas as pd
import gc

sample_rate = 100
MINUTES_AHEAD = 5

gc.collect()


def min_to_sec(minval):
    return minval * 60


def to_sr(a):
    return sample_rate * a

info_columns = ['caseid', 'age', 'sex', 'weight', 'height']

PATH_XTRAIN_NPZ = os.path.join(BASE_DIR, 'x_train.npz')
PATH_YTRAIN_NPZ = os.path.join(BASE_DIR, 'y_train.npz')
PATH_XTRAIN_CSV = os.path.join(BASE_DIR, 'x_train_cases.csv')

if os.path.exists(PATH_XTRAIN_NPZ):
    print("loading train...", flush=True, end='')
    x_train = np.load(PATH_XTRAIN_NPZ)['arr_0']
    y_train = np.load(PATH_YTRAIN_NPZ)['arr_0']
    x_train_cases = pd.read_csv(PATH_XTRAIN_CSV)
    print('done')    
else:
    df_tc = pd.read_csv(FILE_TRAIN_CASES)
    for _, row in df_tc.iterrows():
        caseid = row['caseid']
        age = row['age']
        sex = row['sex']
        weight = row['weight']
        height = row['height']

        info_pack = [caseid, age, sex, weight, height]

        fpath = os.path.join(BASE_DIR, ''.join(['train_data/', str(caseid), '.csv']))
        if os.path.isfile(fpath):
            print(fpath)
            samples = pd.read_csv(fpath, header=None).values.flatten() # 읽은값이 (x, 1) 이어서 1차원으로 바꿔준다.
            print(samples.shape)

            # 20sec (20 00) - 5min (300 00) - 1min (60 00) = 38000 sample

            i = 0
            event_idx = []
            non_event_idx = []
            in_wlen = to_sr(20) # input 사이즈 - 20초 samples
            yidx_start = to_sr(20 + min_to_sec(MINUTES_AHEAD)) # output 크기
            yidx_end = yidx_start + to_sr(min_to_sec(1))
            while i < (len(samples) - yidx_end) :
                # 입력 : +20초간의 MAP
                # 모델 입력으로 쓰인다.
                segx = samples[i : i + in_wlen] # 20sec samples window
                # 출력 : 20초 후부터 5분 후 1분동안의 MAP값
                # 출력값에서 이벤트가 발생했는지 확인해서 모델 출력값을 만든다.
                segy = samples[i + yidx_start : i + yidx_end]

                """
                np.diff()값들 사이의 차이의 배열을 리턴한다. (len(a) - 1 의 길이를 가짐)
                한번에 30 이상의 값 변화가 존재하는 경우 세그먼트가 invalid한 것으로 간주한다.
                """
                
                # 결측값 10% 이상이면
                if np.mean(np.isnan(segx)) > 0.1 or \
                    np.mean(np.isnan(segy)) > 0.1 or \
                    np.max(segx) > 200 or np.min(segx) < 20 or \
                    np.max(segy) > 200 or np.min(segy) < 20 or \
                    np.max(segx) - np.min(segx) < 30 or \
                    np.max(segy) - np.min(segy) < 30 or \
                    (np.abs(np.diff(segx[~np.isnan(segx)])) > 30).any() or \
                    (np.abs(np.diff(segy[~np.isnan(segy)])) > 30).any():
                    i += sample_rate # 1 sec씩 전진
                    continue

                # 출력변수
                segy = moving_average(segy, to_sr(2)) # 2 sec moving average
                #print("caseid=", str(caseid), len(segx), len(segy))
                event = 1 if np.nanmax(segy) < 65 else 0            
                if event:
                    event_idx.append(i)
                    x_train.append(segx)
                    y_train.append(event)
                    x_train_cases.append(info_pack)
                elif np.nanmin(segy) > 65:
                    non_event_idx.append(i)
                    x_train.append(segx)
                    y_train.append(event)
                    x_train_cases.append(info_pack)
                    
                i = i + to_sr(30)

            nsamp = len(event_idx) + len(non_event_idx)
            if nsamp > 0:
                print('{}: {} ({:.1f}%)'.format(caseid, nsamp, len(event_idx) * 100 / nsamp))

    print("saving...", flush=True, end='')

    x_train = np.array(x_train, dtype=np.float32)
    np.savez_compressed(PATH_XTRAIN_NPZ, x_train)

    y_train = np.array(y_train, dtype=np.float32)
    np.savez_compressed(PATH_YTRAIN_NPZ, y_train)

    info_pd = pd.DataFrame(x_train_cases, columns = info_columns)
    info_pd.to_csv(PATH_XTRAIN_CSV, index=False)

    print("done", flush=True)




# test set loading
X_TEST_NPZ = os.path.join(BASE_DIR, "x_test.npz")
if os.path.exists(X_TEST_NPZ):
    print("loading test..." , flush=True, end='')
    x_test = np.load(X_TEST_NPZ, allow_pickle=True)['arr_0']
    print("done")
else:
    x_test = pd.read_csv(os.path.join(BASE_DIR, 'test2_x.csv')).values
    print("saving...", flush=True, end='')
    np.savez_compressed(X_TEST_NPZ, x_test)
    print('done', flush=True)                         


BATCH_SIZE = 512

x_train -= 65
x_train /= 65
x_test -= 65
x_test /= 65


print("fill nan values...train,", flush=True, end='')
# nan을 이전 값으로 채움
x_train = pd.DataFrame(x_train).fillna(method='ffill', axis=1).fillna(method='bfill', axis=1).values
gc.collect()

print("test ", flush=True, end='')
x_test = pd.DataFrame(x_test).fillna(method='ffill', axis=1).fillna(method='bfill', axis=1).values
gc.collect()
print("end")


print(len(x_train))
print(len(y_train))
print(len(x_test))

# change shape for using as input to CNN
print(x_train.shape)
print(x_test.shape)
x_train = x_train[..., None]
x_test = x_test[..., None]

print("train {} ({} events {:.1f}%), test {}".format(len(y_train), sum(y_train), 100 * np.mean(y_train), len(x_test)))

from keras.models import Sequential
from keras models import Model, load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPool1D, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import auc 


