#!/usr/bin/env python
# coding: utf-8

import os, sys, gc
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPool1D, BatchNormalization, Dropout, Activation, Add, Layer, GlobalAveragePooling1D, Input, Concatenate,Reshape, Dense, multiply, add, Permute, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import optimizers
from keras.utils.vis_utils import plot_model
import pandas as pd
import numpy as np
import random

from sklearn import preprocessing
from sklearn.metrics import auc, classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, f1_score, precision_recall_curve
from model import *
from tools import *

BASE_DIR = "./"
FILE_TRAIN_CASES = "./train_cases.csv"        


DATA_VERSION = 14
MINUTES_AHEAD = 5
BATCH_SIZE = 512


def make_filename(typename, ext):
    ret = typename + '_' + str(DATA_VERSION) + "_" + str(random_keyval) + '.' + ext
    ret = os.path.join(BASE_DIR, ret)
    return ret


# extract demographic data from test dataset
if os.path.exists("x_test_cases.csv") == False:
    x_test = pd.read_csv(os.path.join(BASE_DIR, 'test2_x.csv'))    
    x_test_cases = x_test[['age', 'sex', 'weight', 'height']]
    x_test_cases.to_csv("x_test_cases.csv", index=False)

    
def get_scaler():
    """generator에서 데이터 생성할 때마다 사용할 scaler를 생성함
        
        치우침이 없도록 test 데이터도 포함해서 scaler를 fit 한다.
    """    
    x_train_cases = pd.read_csv(FILE_TRAIN_CASES)
    x_train_cases = x_train_cases.drop('caseid', axis=1)
    x_test_cases = pd.read_csv("x_test_cases.csv")
    tmp = pd.concat([x_train_cases, x_test_cases])    
    tmp = pd.get_dummies(tmp)
    scaler = preprocessing.MinMaxScaler()    
    tmp = scaler.fit_transform(tmp)        
    return scaler


demographic_info_scaler = get_scaler()

def build_one_epoch(scaler, frac=0.5):
    gc.collect()

    x_train = []
    x_train_cases = []
    y_train = []
    info_columns = ['caseid', 'age', 'sex', 'weight', 'height']
    hb_err_cnt = 0

    df_tc = pd.read_csv(FILE_TRAIN_CASES)    
    df_tc = df_tc.sample(frac=frac)
        
    num_processed = 0
    for idx, row in df_tc.iterrows():
        caseid = row['caseid']
        age = row['age']
        sex = row['sex']
        weight = row['weight']
        height = row['height']
        
        num_processed += 1
        
        b = "Building 1-epoch Data " + str(int((num_processed * 100)/df_tc.shape[0])) + "%"
        sys.stdout.write('\r'+ b)

        fpath = os.path.join(BASE_DIR, ''.join(['train_data/', str(caseid), '.csv']))
        if os.path.isfile(fpath):
            samples = pd.read_csv(fpath, header=None).values.flatten() # 읽은값이 (x, 1) 이어서 1차원으로 바꿔준다.

            # 20sec (20 00) - 5min (300 00) - 1min (60 00) = 38000 sample
            
            i = random.randint(to_sr(10), to_sr(40)) # random start

            event_idx = []
            non_event_idx = []

            yidx_start = to_sr(20 + min_to_sec(MINUTES_AHEAD)) # output 크기
            yidx_end = yidx_start + to_sr(min_to_sec(1))
            while i < (len(samples) - yidx_end) :                
                segx = samples[i : i + in_wlen] # 20sec samples window
                segy = samples[i + yidx_start : i + yidx_end] # 출력 : 20초 후부터 5분 후 1분동안의 MAP값
                
                if (check_valid(segx) == False) or (check_valid(segy) == False):
                    i += random.randint(to_sr(1), to_sr(10)) # random move
                    continue

                segx = fb_fill_1dim(segx) # nan 값을 앞뒤 값으로 채운다.

                # demographic data 패키징
                info_pack = [caseid, age, sex, weight, height]

                # 출력변수
                segy = moving_average(segy, to_sr(2)) # 2 sec moving average

                # event 여부 확인
                event = 1 if np.nanmax(segy) < 65 else 0

                # train 데이터 추가
                event_idx.append(i) if event else non_event_idx.append(i)
                x_train.append(segx)
                y_train.append(event)
                x_train_cases.append(info_pack)
                
                # move i by random
                i += random.randint(to_sr(5), to_sr(35))

            nsamp = len(event_idx) + len(non_event_idx)

    print()
    x_train = np.array(x_train, dtype=np.float32)
    
    # postprocess x_train value if needed
    x_train -= 65
    x_train /= 65

    y_train = np.array(y_train, dtype=np.float32)
    
    # one-hot encode categorical columns and normalize.
    x_train_cases = pd.DataFrame(x_train_cases, columns = info_columns)
    gc.collect()
    _xtrain_cases = x_train_cases
    if 'caseid' in _xtrain_cases:
        _xtrain_cases = _xtrain_cases.drop('caseid', axis=1)

    _xtest_cases = pd.read_csv(os.path.join(BASE_DIR, "x_test_cases.csv"))
    tmp = pd.concat([_xtrain_cases, _xtest_cases])
    _xtrain_cases_len = x_train_cases.shape[0]
    tmp = pd.get_dummies(tmp)
    tmp = scaler.transform(tmp)    
    xtrain_cases = tmp[:_xtrain_cases_len]    

    # shuffle data several times
    shuffle_cnt = random.randint(4, 10)
    for i in range(shuffle_cnt):
        s = np.arange(x_train.shape[0])
        np.random.shuffle(s)
        x_train = x_train[s]
        y_train = y_train[s]
        xtrain_cases = xtrain_cases[s]
    
    # concatenate MBP and demographic data
    # will be split in first layer of model
    x_train = np.concatenate([x_train, xtrain_cases], axis=1)    
    return x_train, y_train


class Generator(tf.keras.utils.Sequence):
    """data generator for """
    def __init__(self, batch_size=256, frac=0.05):
        self.batch_size = batch_size
        self.frac = frac
        self.on_epoch_end()

    def __len__(self): # return number of batches
        #print(self.x_train.shape[0], self.batch_size)
        num_batch = int(self.x_train.shape[0] / self.batch_size)
        if self.x_train.shape[0] % self.batch_size > 0:
            num_batch += 1
        #print("num batch=", num_batch)
        return num_batch

    def __getitem__(self, idx): # return 1-batch data
        start = idx * self.batch_size
        end = (idx+1) * self.batch_size
        if end > len(self.x_train):
            end = len(self.x_train)
        
        batch_x = self.x_train[start:end]
        batch_y = self.y_train[start:end]
        return batch_x, batch_y

    def on_epoch_end(self):
        x_train, y_train = build_one_epoch(scaler=demographic_info_scaler, frac=self.frac)
        self.x_train = x_train        
        self.y_train = y_train


input_tensor = Input(shape=(2005,), dtype='float32')

kernel_sizes = [3]
filter_sizes = [32, 32, 32, 64, 64, 64, 128, 128]

def prepare_model():
    tf.keras.backend.clear_session()
    mergenet = resnet_demo_net(input_tensor, kernel_sizes=kernel_sizes, filter_sizes=filter_sizes)
    mergenet_model = keras.models.Model(inputs=input_tensor, outputs=mergenet)
    mergenet_model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy', tf.keras.metrics.AUC(name="auc_mergenet")])
    return mergenet_model

mergenet_model = prepare_model()


EPOCHS = 1
BATCH_SIZE = 256

def make_callbacks(weight_path):
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='loss', filepath=weight_path, verbose=1, save_best_only=True))
    #callbacks.append(ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience=2, min_lr=0.00001, verbose=1, mode='min'))    
    callbacks.append(EarlyStopping(monitor='val_auc_mergenet', patience=1, verbose=0, mode='max', min_delta=0.015))
    return callbacks


def train_model(model, weight_path, saving_model_path, batch_size, epochs):    

    callbacks = make_callbacks(weight_path)
    
    train_gen = Generator(BATCH_SIZE, frac=0.01)
    valid_gen = Generator(BATCH_SIZE, frac=0.01)
    
    hist = model.fit(train_gen, validation_data=valid_gen, batch_size=batch_size, class_weight={0:1, 1:10}, callbacks=callbacks, epochs=epochs)
    
    open(saving_model_path, "wt").write(model.to_json())

    return model


# model/weight name
model_ver = 6
model_name = "mergenet_model"
outdir = "se_resnet_with_generator" + "_" + str(model_ver)
weight_path = os.path.join(BASE_DIR, outdir, model_name + "weights.hdf5")
os.makedirs(os.path.dirname(weight_path), exist_ok=True)
saving_model_path = os.path.join(BASE_DIR, outdir, model_name + "model.json")


# train model :
model = train_model(mergenet_model, weight_path, saving_model_path, BATCH_SIZE, EPOCHS)


# prepare test data
x_test = []
x_test_cases = None
x_test = pd.read_csv(os.path.join(BASE_DIR, 'test2_x.csv'))    
x_test = x_test.values
x_test = np.array(x_test[:, 4:], dtype=np.float32) # skip demographic values(age, sex weight, height)
x_test = ffill(x_test) # fill nan with valid prev
x_test = bfill(x_test) # fill nan with valid next
x_test -= 65
x_test /= 65


# predict submit data
x_test_cases = pd.read_csv(os.path.join(BASE_DIR, "x_test_cases.csv"))
x_test_cases = pd.get_dummies(x_test_cases)
xtest_cases = demographic_info_scaler.transform(x_test_cases)
x_test_input = np.concatenate([x_test, xtest_cases], axis=1)

y_pred = model.predict(x = x_test_input).flatten()

def savepredfile(filename, pred):
    savefilename = os.path.join(BASE_DIR, "preds")
    savefilename = os.path.join(savefilename, filename)
    np.savetxt(savefilename, pred, fmt="%.4f")
    np.savetxt(filename, pred, fmt="%.4f")

savepredfile("pred_y.txt", y_pred)

