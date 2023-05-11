from torch.utils.data import Dataset
from transformers import AutoTokenizer, LlamaTokenizer, LlamaModel, BertModel
from sentence_transformers import SentenceTransformer
from ast import literal_eval
import pandas as pd
import torch

class Datasetor(Dataset):
    def __init__(self, csv,
                 model_name,
                 sentence_bert_name,
                 win_len,
                 step_len,
                 max_token_num,
                 bbox_flag,
                 sentence_bert_flag,
                 device):
        self.csv = csv
        self.device = device
        self.bbox_flag = bbox_flag
        self.sentence_bert_flag = sentence_bert_flag
        self.max_token_num = max_token_num
        self.sentence_bert = SentenceTransformer(sentence_bert_name)
        if 'llama' in model_name:
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = '<pad>'
            self.model = LlamaModel.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)

        self.sentence, self.label_group, self.bbox = \
            self.bulk_data(win_len=win_len, step_len=step_len)

    def flat_data(self):
        sentence = []
        label_group = []
        bbox = []
        for index in range(len(self.csv)):
            sentence += literal_eval(self.csv['sentence'][index])
            label_group += literal_eval(self.csv['label_group'][index])
            if self.bbox_flag:
                bbox += literal_eval(self.csv['bbox'][index])

        return sentence, label_group, bbox

    def bulk_data(self, win_len, step_len):
        sentence, label_group, bbox= self.flat_data()
        result_sentence = []
        result_label_group = []
        result_bbox = []
        for i in range(0, len(sentence) - win_len + 1, step_len):
            result_sentence.append(sentence[i:i + win_len])
            result_label_group.append(label_group[i:i + win_len])
            if self.bbox_flag:
                result_bbox.append(bbox[i:i + win_len])

        return result_sentence, result_label_group, result_bbox

    def __len__(self):
        return len(self.sentence)

    def get_tokens(self, sentence_list):
        tokens = self.tokenizer(sentence_list,
                                max_length=self.max_token_num,
                                truncation=True,
                                padding='max_length',
                                return_tensors='pt')
        return tokens

    def get_labels(self, label_group_list):
        label_cos_sim_matrix = [[0] * len(label_group_list)
                                for i in range(len(label_group_list))]
        label_cos_sim_list = []
        label_seg = []
        label_fig = []
        for index_1 in range(len(label_group_list)):
            for index_2 in range(index_1):
                if label_group_list[index_1] == label_group_list[index_2]:
                    label_fig.append(1)
                else:
                    label_fig.append(0)

        for index in range(len(label_group_list)-1):
            if label_group_list[index] == label_group_list[index+1]:
                label_seg.append(0)
            else:
                label_seg.append(1)

        for index_1 in range(len(label_group_list)):
            for index_2 in range(index_1, len(label_group_list)):
                if label_group_list[index_1] == label_group_list[index_2]:
                    label_cos_sim_list.append(1)
                else:
                    label_cos_sim_list.append(0)

        for index_1 in range(len(label_group_list)):
            for index_2 in range(len(label_group_list)):
                if label_group_list[index_1] == label_group_list[index_2]:
                    label_cos_sim_matrix[index_1][index_2] = 1
                else:
                    label_cos_sim_matrix[index_1][index_2] = 0

        return torch.tensor(label_seg).to(self.device), \
               torch.tensor(label_cos_sim_list).to(self.device), \
               torch.tensor(label_cos_sim_matrix).to(self.device), \
               torch.tensor(label_fig).to(self.device)

    def get_bbox(self, bbox_list):
        bbox = [v[0] + v[2] for v in bbox_list]
        for bbox_sub in bbox:
            for index in range(len(bbox_sub)):
                if bbox_sub[index] > 5000:
                    bbox_sub[index] = 4990
                bbox_sub[index] = int(1000 * (bbox_sub[index] / 5000))

        return bbox

    def __getitem__(self, item):
        sentence_list = self.sentence[item]
        label_group_list = self.label_group[item]
        tokens = self.get_tokens(sentence_list)
        label_seg, label_cos_sim_list, \
        label_cos_sim_matrix, label_fig = self.get_labels(label_group_list)
        if self.sentence_bert_flag:
            sentence_bert_vec = self.sentence_bert.encode(sentence_list)
            sentence_bert_vec = torch.tensor(sentence_bert_vec).to(self.device)
        else:
            sentence_bert_vec = -100

        if self.bbox_flag:
            bbox_list = self.bbox[item]
            bbox = torch.tensor(self.get_bbox(bbox_list)).to(self.device)
        else:
            bbox = -100

        result = {'input_ids': tokens['input_ids'].to(self.device),
                  'attention_mask': tokens['attention_mask'].to(self.device),
                  'label_seg': label_seg,
                  'label_fig': label_fig,
                  'label_cos_sim_list': label_cos_sim_list,
                  'label_cos_sim_matrix': label_cos_sim_matrix,
                  'sentence_bert_vec': sentence_bert_vec,
                  'bbox': bbox
                  }
        return result

if __name__ == "__main__":
    csv = pd.read_csv('../data/train_fr.csv')
    obj = Datasetor(csv=csv,
                    model_name='camembert-base',
                    sentence_bert_name='sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens',
                    win_len=2,
                    step_len=2,
                    max_token_num=512,
                    bbox_flag=True,
                    sentence_bert_flag=True,
                    device='cuda:0')
    print(obj[4])




