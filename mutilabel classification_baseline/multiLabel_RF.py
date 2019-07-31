import pandas as pd
import numpy as np
import logging
import os, sys

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import *
#from sklearn.cross_validation import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

DATA_ROOT = '/ifs/data1/wangjiacheng/2step2/goa/'
TRAIN_DATA_ROOT = '/ifs/data1/wangjiacheng/2step3/train_data/'
TEST_DATA_ROOT = '/ifs/data1/wangjiacheng/2step3/test_data/'
FUNCTION = 'bp'

func_df = pd.read_pickle(DATA_ROOT + FUNCTION + '.pkl')
global functions
functions = func_df['functions'].values
n_class = len(functions)

df = pd.read_pickle(TRAIN_DATA_ROOT + 'train' + '-line_emb_s100_n10-' + FUNCTION + '.pkl')
test_df = pd.read_pickle(TEST_DATA_ROOT + 'test' + '-line_emb_s100_n10-' + FUNCTION + '.pkl')
n = len(df)
index = df.index.values
validat_n = int(n * 0.8)
#train_df = df.loc[index[:validat_n]]
#validat_df = df.loc[index[validat_n:]]

def reshape(values):
    values = np.hstack(values).reshape(
    	len(values), len(values[0]))
    return values

def get_values(data_frame):
    labels = reshape(data_frame['labels'].values)
    rep = reshape(data_frame['embeddings'].values)
    return rep, labels

train = get_values(df)
#validat = get_values(validat_df)
test = get_values(test_df)

train_rep, train_labels = train
#validat_rep, validat_labels = validat
test_rep, test_labels = test

params = {
	'max_depth': 13,
    #'n_estimators': 200,
    #'min_samples_leaf': 10,
    #'min_samples_split': 2,
    'n_jobs': -1,
    'random_state': 0
	#'random_stat': 2
	#'n_classes_': n_class,
}

cls = RandomForestClassifier(**params)
cls.fit(train_rep, train_labels)
y_pred_pro = cls.predict_proba(test_rep)
y_pred = cls.predict(test_rep)
#y_pred = y_pred_pro.argmax(axis = 1)
confusion_matrix(test_labels.argmax(axis=1), y_pred.argmax(axis=1))

precision_score(test_labels, y_pred, average = 'samples')
recall_score(test_labels, y_pred, average = 'samples')
f1_score(test_labels, y_pred, average = 'samples')

print(classification_report(test_labels, y_pred))

fr = open(TEST_DATA_ROOT +'multiLabel_RF_dep13_result_samples_' + FUNCTION + '.txt', 'w')
for i in range(len(y_pred)):
    fr.write(str(y_pred[i]) + '\t' + str(y_pred_pro[i][1]) + '\n')
fr.close()