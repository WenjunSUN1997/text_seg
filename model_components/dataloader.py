from torch.utils.data import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from model_components.datasetor import Datasetor
import pandas as pd

def get_dataloader(dataset_name,
                   model_name,
                   sentence_bert_name,
                   win_len,
                   step_len,
                   max_token_num,
                   bbox_flag,
                   sentence_bert_flag,
                   device,
                   goal,
                   batch_size):
    dataset_name_dict = {'choi':{'train': 'data/train_choi.csv',
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
    csv = pd.read_csv(dataset_name_dict[dataset_name][goal])
    dataset = Datasetor(csv,
                        model_name,
                        sentence_bert_name,
                        win_len,
                        step_len,
                        max_token_num,
                        bbox_flag,
                        sentence_bert_flag,
                        device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

