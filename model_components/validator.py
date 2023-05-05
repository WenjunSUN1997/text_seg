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
    for data in tqdm(dataloader):
        output = seg_model(data)
        prediction_all += output['seg_result']
        label_all += data['label_seg'].H.cpu().tolist()[0]

    pk = nltk.pk(ref=label_all, hyp=prediction_all, boundary=1)
    window_diff = nltk.windowdiff(seg1=label_all, seg2=prediction_all, boundary=1)
    return_value = {'pk': pk,
                    'window_diff': window_diff,}
    return return_value