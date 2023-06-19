import pandas as pd
from ast import literal_eval

def static(csv):
    sentence_num = 0
    article_num = 0
    for index in range(len(csv)):
        sentence_num += len(literal_eval(csv['sentence'][index]))
        article_num += len(literal_eval(csv['label_seg'][index])) - \
                       sum(literal_eval(csv['label_seg'][index]))
        # print(len(literal_eval(csv['sentence'][index])))
        # print(sum(literal_eval(csv['label_seg'][index])))
    return sentence_num, article_num

if __name__ == "__main__":
    dataset_name_dict = {'choi': {'train': 'data/train_choi.csv',
                                  'dev': 'data/dev_choi.csv',
                                  'val': 'data/val_choi.csv'},
                         '50': {'train': 'data/train_wiki50.csv',
                                'dev': 'data/dev_wiki50.csv',
                                'val': 'data/val_wiki50.csv'},
                         'fr': {'train': 'data/train_fr.csv',
                                'dev': 'data/dev_fr.csv',
                                'val': 'data/val_fr.csv'},
                         'fi': {'train': 'data/train_fi.csv',
                                'dev': 'data/dev_fi.csv',
                                'val': 'data/val_fi.csv'},
                         'city': {'train': 'data/train_wikicity.csv',
                                  'dev': 'data/dev_wikicity.csv',
                                  'val': 'data/val_wikicity.csv'},
                         'diseases': {'train': 'data/train_wikidiseases.csv',
                                      'dev': 'data/dev_wikidiseases.csv',
                                      'val': 'data/val_wikidiseases.csv'},
                         }
    goal_list = ['train', 'dev', 'val']
    for dataset_name in dataset_name_dict.keys():
        print(dataset_name)
        sentence_num = 0
        article_num = 0
        for goal in goal_list:
            path = '../' + dataset_name_dict[dataset_name][goal]
            csv = pd.read_csv(path)
            sentence_num += static(csv)[0]
            article_num += static(csv)[1]

        print(dataset_name)
        print(sentence_num)
        print(article_num)
        print(sentence_num / article_num)





