from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from sklearn.metrics import confusion_matrix
import pandas as pd
import os
import nltk

def validate(seg_model, dataloader, loss_func):
    print('val')
    loss_all = []
    prediction_all = []
    label_all = []
    labels_to_cal = [1]
    for data in tqdm(dataloader):
        output = seg_model(data)
        prediction_all += output['seg_result']
        label_all += data['label_seg'].H.cpu().tolist()[0]

    label_all_nltk = convert_seg_to_nltk(label_all)
    prediction_all_nltk = convert_seg_to_nltk(prediction_all)
    pk = nltk.pk(ref=label_all_nltk, hyp=prediction_all_nltk)
    p = precision_score(y_true=label_all, y_pred=prediction_all,
                        average='micro', labels=labels_to_cal)
    r = recall_score(y_true=label_all, y_pred=prediction_all,
                        average='micro', labels=labels_to_cal)

    return_value = {'pk': pk,
                    'p': p,
                    'r': r,
                    'loss': 0}
    return return_value

def convert_seg_to_nltk(seg_result):
    nltk_format = ''
    for seg in seg_result:
        if seg == 0:
            nltk_format += '0'
        else:
            nltk_format += '1'

    return nltk_format