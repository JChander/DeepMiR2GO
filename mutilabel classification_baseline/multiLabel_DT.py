import pandas as pd
import numpy as np
import logging
import os, sys
from collections import deque
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import *
#from sklearn.cross_validation import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier

DATA_ROOT = '/ifs/data1/wangjiacheng/2step2/goa/'
TRAIN_DATA_ROOT = '/ifs/data1/wangjiacheng/2step3/train_data/'
TEST_DATA_ROOT = '/ifs/data1/wangjiacheng/2step3/test_data/'
FUNCTION = 'bp'

def main():
    global go
    go = get_gene_ontology('go.obo')
    func_df = pd.read_pickle(DATA_ROOT + FUNCTION + '.pkl')
    global functions
    functions = func_df['functions'].values
    n_class = len(functions)

    df = pd.read_pickle(TRAIN_DATA_ROOT + 'train' + '-line_emb_s100_n10-' + FUNCTION + '.pkl')
    test_df = pd.read_pickle(TEST_DATA_ROOT + 'test' + '-line_emb_s100_n10-' + FUNCTION + '.pkl')
    n = len(df)
    index = df.index.values
    validat_n = int(n * 0.8)

    train = get_values(df)
#validat = get_values(validat_df)
    test = get_values(test_df)
    test_gos = test_df['gos'].values
    train_rep, train_labels = train
#validat_rep, validat_labels = validat
    test_rep, test_labels = test
    print(len(test_labels))

    params = {
        'max_depth': 13,
        #'random_stat': 2
        #'n_classes_': n_class,
    }

    cls = DecisionTreeClassifier(**params)
    cls.fit(train_rep, train_labels)
    y_pred_pro = cls.predict_proba(test_rep)
    y_pred = cls.predict(test_rep)
    #y_pred = y_pred_pro.argmax(axis = 1)
    #confusion_matrix(test_labels.argmax(axis=1), y_pred.argmax(axis=1))

    #precision_score(test_labels, y_pred, average = 'samples')
    #recall_score(test_labels, y_pred, average = 'samples')
    #f1_score(test_labels, y_pred, average = 'samples')

    f, p, r, t, preds_max = compute_performance(y_pred_pro, test_labels, test_gos)
    logging.info('Fmax measure: \t %f %f %f %f' % (f, p, r, t))
        #logging.info('ROC AUC: \t %f ' % (roc_auc, ))
        #logging.info('MCC: \t %f ' % (mcc, ))
    print('%.3f & %.3f & %.3f' % (
           f, p, r))

def reshape(values):
    values = np.hstack(values).reshape(
    	len(values), len(values[0]))
    return values

def get_values(data_frame):
    labels = reshape(data_frame['labels'].values)
    rep = reshape(data_frame['embeddings'].values)
    #gos = reshape(data_frame['gos'].values)
    return rep, labels

def get_gene_ontology(filename='go.obo'):
    # Reading Gene Ontology from OBO Formatted file
    go = dict()
    obj = None
    with open('/ifs/data1/wangjiacheng/2step2/goa/' + filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == '[Term]':
                if obj is not None:
                    go[obj['id']] = obj
                obj = dict()
                obj['is_a'] = list()
                obj['part_of'] = list()
                obj['regulates'] = list()
                obj['is_obsolete'] = False
                continue
            elif line == '[Typedef]':
                obj = None
            else:
                if obj is None:
                    continue
                l = line.split(": ")
                if l[0] == 'id':
                    obj['id'] = l[1]
                elif l[0] == 'is_a':
                    obj['is_a'].append(l[1].split(' ! ')[0])
                elif l[0] == 'name':
                    obj['name'] = l[1]
                elif l[0] == 'is_obsolete' and l[1] == 'true':
                    obj['is_obsolete'] = True
    if obj is not None:
        go[obj['id']] = obj
    for go_id in list(go.keys()):
        if go[go_id]['is_obsolete']:
            del go[go_id]
    for go_id, val in go.items():
        if 'children' not in val:
            val['children'] = set()
        for p_id in val['is_a']:
            if p_id in go:
                if 'children' not in go[p_id]:
                    go[p_id]['children'] = set()
                go[p_id]['children'].add(go_id)
    return go

def get_anchestors(go, go_id):
    go_set = set()
    q = deque()
    q.append(go_id)
    while(len(q) > 0):
        g_id = q.popleft()
        go_set.add(g_id)
        for parent_id in go[g_id]['is_a']:
            if parent_id in go:
                q.append(parent_id)
    return go_set

def compute_performance(preds, labels, gos):
    preds = np.round(preds, 2)
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        total = 0
        f = 0.0
        p = 0.0
        r = 0.0
        p_total = 0
        for i in range(labels.shape[0]):
            tp = np.sum(predictions[i, :] * labels[i, :])
            fp = np.sum(predictions[i, :]) - tp
            fn = np.sum(labels[i, :]) - tp
            all_gos = set()
            for go_id in gos[i]:
                if go_id in all_functions:
                    all_gos |= get_anchestors(go, go_id)
            all_gos.discard(GO_ID)
            all_gos -= func_set
            fn += len(all_gos)
            if tp == 0 and fp == 0 and fn == 0:
                continue
            total += 1
            if tp != 0:
                p_total += 1
                precision = tp / (1.0 * (tp + fp))
                recall = tp / (1.0 * (tp + fn))
                p += precision
                r += recall
        if p_total == 0:
            continue
        r /= total
        p /= p_total
        if p + r > 0:
            f = 2 * p * r / (p + r)
            if f_max < f:
                f_max = f
                p_max = p
                r_max = r
                t_max = threshold
                predictions_max = predictions
    return f_max, p_max, r_max, t_max, predictions_max



#print(classification_report(test_labels, y_pred))

#fr = open(TEST_DATA_ROOT +'multiLabel_DT_dep13_result_samples_' + FUNCTION + '.txt', 'w')
#for i in range(len(y_pred)):
#    fr.write(str(y_pred[i]) + '\t' + str(y_pred_pro[i][1]) + '\n')

#fr.close()

if __name__ == '__main__':
    main()
