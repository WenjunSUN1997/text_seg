import os
from datasets import load_dataset
import pandas as pd
from xml_processor import XmlProcessor
from tqdm import tqdm
from ast import literal_eval

def process_wiki727():
    dataset = load_dataset('laiviet/wiki_727')
    df = dataset['test']
    df = pd.DataFrame(df)

def process_choi():
    sentences = []
    label_seg = []
    label_group = []
    label_cos_sim = []
    path = '../data/choi/data/choi/'
    path_list = []
    folder_index = ['1/', '2/', '3/', '4/']
    for index in folder_index:
        folder_index_sub = os.listdir(path+index)
        for index_sub in folder_index_sub:
            folder_index_sub_sub = os.listdir(path + index + index_sub)
            for index_sub_sub in folder_index_sub_sub:
                path_list.append(path + index + index_sub + '/' + index_sub_sub)

    for file_path in path_list:
        with open(file_path, 'r') as file:
            text = file.read()
        groups = [i for i in text.split('==========') if i != '']
        sentences_this_article = []
        label_group_this_article = []
        for group_index, sub_group in enumerate(groups):
            sentences_list = [i for i in sub_group.split('\n') if i != '']
            sentences_this_article += sentences_list
            label_group_this_article += [group_index] * len(sentences_list)

        label_group.append(label_group_this_article)
        label_seg_this_article = []
        label_cos_sim_this_article = []
        for index in range(len(label_group_this_article)-1):
            if label_group_this_article[index] == label_group_this_article[index+1]:
                label_seg_this_article.append(1)
            else:
                label_seg_this_article.append(0)
        label_seg.append(label_seg_this_article)

        for index in range(len(label_group_this_article)):
            for index_next in range(index, len(label_group_this_article)):
                if label_group_this_article[index] == label_group_this_article[index_next]:
                    label_cos_sim_this_article.append(1)
                else:
                    label_cos_sim_this_article.append(0)
        label_cos_sim.append(label_cos_sim_this_article)

        sentences.append(sentences_this_article)

    dataframe = pd.DataFrame({'sentence': sentences,
                              'label_seg': label_seg,
                              'label_cos_sim': label_cos_sim,
                              'label_group': label_group})
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    df_train = dataframe.sample(frac=0.8, random_state=123)
    df_dev = dataframe.drop(df_train.index).sample(frac=0.5, random_state=456)
    df_val = dataframe.drop(df_train.index).drop(df_dev.index)
    df_train = df_train.reset_index(drop=True)
    df_dev = df_dev.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_train.to_csv('../data/train_choi.csv')
    df_dev.to_csv('../data/dev_choi.csv')
    df_val.to_csv('../data/val_choi.csv')

def process_wiki50():
    sentences = []
    label_seg = []
    label_group = []
    label_cos_sim = []
    path = '../data/wiki_50/'
    path_list = []
    folder_index = os.listdir(path)
    for index in folder_index:
        path_list.append(path + index)

    for file_path in path_list:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        groups = [i for i in text.split('========') if i != '']
        sentences_this_article = []
        label_group_this_article = []
        for group_index, sub_group in enumerate(groups):
            sentences_list = [i for i in sub_group.split('\n') if i != '']
            sentences_list = [i for i in sentences_list if '***LIST***' not in i]
            sentences_this_article += sentences_list
            label_group_this_article += [group_index] * len(sentences_list)

        label_group.append(label_group_this_article)
        label_seg_this_article = []
        label_cos_sim_this_article = []
        for index in range(len(label_group_this_article) - 1):
            if label_group_this_article[index] == label_group_this_article[index + 1]:
                label_seg_this_article.append(1)
            else:
                label_seg_this_article.append(0)
        label_seg.append(label_seg_this_article)

        for index in range(len(label_group_this_article)):
            for index_next in range(index, len(label_group_this_article)):
                if label_group_this_article[index] == label_group_this_article[index_next]:
                    label_cos_sim_this_article.append(1)
                else:
                    label_cos_sim_this_article.append(0)
        label_cos_sim.append(label_cos_sim_this_article)

        sentences.append(sentences_this_article)

    dataframe = pd.DataFrame({'sentence': sentences,
                              'label_seg': label_seg,
                              'label_cos_sim': label_cos_sim,
                              'label_group': label_group})
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    df_train = dataframe.sample(frac=0.8, random_state=123)
    df_dev = dataframe.drop(df_train.index).sample(frac=0.5, random_state=456)
    df_val = dataframe.drop(df_train.index).drop(df_dev.index)
    df_train = df_train.reset_index(drop=True)
    df_dev = df_dev.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_train.to_csv('../data/train_wiki50.csv')
    df_dev.to_csv('../data/dev_wiki50.csv')
    df_val.to_csv('../data/val_wiki50.csv')

def process_wiki_city():
    sentences = []
    label_seg = []
    label_group = []
    label_cos_sim = []
    path = '../data/wiki_city/en_city_train/'
    path_list = []
    folder_index = os.listdir(path)
    for index in folder_index:
        path_list.append(path + index)

    for file_path in path_list:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        groups = [i for i in text.split('==========') if i != '']
        sentences_this_article = []
        label_group_this_article = []
        for group_index, sub_group in enumerate(groups):
            sentences_list = [i for i in sub_group.split('\n') if i != '']
            sentences_list = [i for i in sentences_list if i != ';']
            sentences_this_article += sentences_list
            label_group_this_article += [group_index] * len(sentences_list)

        label_group.append(label_group_this_article)
        label_seg_this_article = []
        label_cos_sim_this_article = []
        for index in range(len(label_group_this_article) - 1):
            if label_group_this_article[index] == label_group_this_article[index + 1]:
                label_seg_this_article.append(1)
            else:
                label_seg_this_article.append(0)
        label_seg.append(label_seg_this_article)

        for index in range(len(label_group_this_article)):
            for index_next in range(index, len(label_group_this_article)):
                if label_group_this_article[index] == label_group_this_article[index_next]:
                    label_cos_sim_this_article.append(1)
                else:
                    label_cos_sim_this_article.append(0)
        label_cos_sim.append(label_cos_sim_this_article)

        sentences.append(sentences_this_article)

    dataframe = pd.DataFrame({'sentence': sentences,
                              'label_seg': label_seg,
                              'label_cos_sim': label_cos_sim,
                              'label_group': label_group})
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    # df_train = dataframe.sample(frac=0.8, random_state=123)
    # df_dev = dataframe.drop(df_train.index).sample(frac=0.5, random_state=456)
    # df_val = dataframe.drop(df_train.index).drop(df_dev.index)
    df_train = dataframe.reset_index(drop=True)
    # df_dev = df_dev.reset_index(drop=True)
    # df_val = df_val.reset_index(drop=True)
    df_train.to_csv('../data/train_wikicity.csv')
    # df_dev.to_csv('../data/dev_wiki50.csv')
    # df_val.to_csv('../data/val_wiki50.csv')

def process_wiki_diseases():
    sentences = []
    label_seg = []
    label_group = []
    label_cos_sim = []
    path = '../data/wiki_diseases/en_disease_validation/'
    path_list = []
    folder_index = os.listdir(path)
    for index in folder_index:
        path_list.append(path + index)

    for file_path in path_list:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        groups = [i for i in text.split('==========') if i != '']
        sentences_this_article = []
        label_group_this_article = []
        for group_index, sub_group in enumerate(groups):
            sentences_list = [i for i in sub_group.split('\n') if i != '']
            sentences_list = [i for i in sentences_list if i != ';']
            sentences_this_article += sentences_list
            label_group_this_article += [group_index] * len(sentences_list)

        label_group.append(label_group_this_article)
        label_seg_this_article = []
        label_cos_sim_this_article = []
        for index in range(len(label_group_this_article) - 1):
            if label_group_this_article[index] == label_group_this_article[index + 1]:
                label_seg_this_article.append(1)
            else:
                label_seg_this_article.append(0)
        label_seg.append(label_seg_this_article)

        for index in range(len(label_group_this_article)):
            for index_next in range(index, len(label_group_this_article)):
                if label_group_this_article[index] == label_group_this_article[index_next]:
                    label_cos_sim_this_article.append(1)
                else:
                    label_cos_sim_this_article.append(0)
        label_cos_sim.append(label_cos_sim_this_article)

        sentences.append(sentences_this_article)

    dataframe = pd.DataFrame({'sentence': sentences,
                              'label_seg': label_seg,
                              'label_cos_sim': label_cos_sim,
                              'label_group': label_group})
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    # df_train = dataframe.sample(frac=0.8, random_state=123)
    # df_dev = dataframe.drop(df_train.index).sample(frac=0.5, random_state=456)
    # df_val = dataframe.drop(df_train.index).drop(df_dev.index)
    df_train = dataframe.reset_index(drop=True)
    # df_dev = df_dev.reset_index(drop=True)
    # df_val = df_val.reset_index(drop=True)
    df_train.to_csv('../data/val_wikidiseases.csv')
    # df_dev.to_csv('../data/dev_wiki50.csv')
    # df_val.to_csv('../data/val_wiki50.csv')

def process_newseye():
    path = '../data/fi/'
    sentences = []
    bbox = []
    label_seg = []
    label_group = []
    label_cos_sim = []
    folder_index = os.listdir(path)
    for file in tqdm(folder_index):
        xml_obj = XmlProcessor(1, path + file)
        annotation_list = xml_obj.get_annotation()
        sentence_this_article = []
        bbox_this_article = []
        label_group_this_article = []
        label_seg_this_article = []
        label_cos_sim_this_article = []
        for annotation in annotation_list:
            sentence_this_article.append(annotation['text'])
            bbox_this_article.append(annotation['bbox'])
            label_group_this_article.append(int(annotation['reading_order'].
                                                split('a')[-1]))
        for index in range(len(label_group_this_article) - 1):
            if label_group_this_article[index] == label_group_this_article[index + 1]:
                label_seg_this_article.append(1)
            else:
                label_seg_this_article.append(0)

        for index in range(len(label_group_this_article)):
            for index_next in range(index, len(label_group_this_article)):
                if label_group_this_article[index] == label_group_this_article[index_next]:
                    label_cos_sim_this_article.append(1)
                else:
                    label_cos_sim_this_article.append(0)

        label_seg.append(label_seg_this_article)
        label_group.append(label_group_this_article)
        label_cos_sim.append(label_cos_sim_this_article)
        sentences.append(sentence_this_article)
        bbox.append(bbox_this_article)

    dataframe = pd.DataFrame({'sentence': sentences,
                              'label_seg': label_seg,
                              'label_cos_sim': label_cos_sim,
                              'label_group': label_group,
                              'bbox': bbox})
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    df_train = dataframe.sample(frac=0.8, random_state=123)
    df_dev = dataframe.drop(df_train.index).sample(frac=0.5, random_state=456)
    df_val = dataframe.drop(df_train.index).drop(df_dev.index)
    df_train = df_train.reset_index(drop=True)
    df_dev = df_dev.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_train.to_csv('../data/train_fi.csv')
    df_dev.to_csv('../data/dev_fi.csv')
    df_val.to_csv('../data/val_fi.csv')

def reset_group_label():
    dataset_name = ['choi', 'fi', 'fr', 'wiki50', 'wikicity', 'wikidiseases']
    type = ['train_', 'val_', 'dev_']
    path = '../data/'
    for type_sub in type:
        for dataset_name_sub in dataset_name:
            df = pd.read_csv(path+type_sub+dataset_name_sub+'.csv')
            new_label_group = []
            for index in range(len(df['label_group'])):
                label = literal_eval(df['label_group'][index])
                for label_index in range(len(label)):
                    label[label_index] = str(index) + '_'+ str(label[label_index])
                new_label_group.append(label)
            df['label_group'] = new_label_group
            df.to_csv(path+type_sub+dataset_name_sub+'.csv')


if __name__ == "__main__":
    reset_group_label()