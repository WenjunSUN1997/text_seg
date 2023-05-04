from torch.utils.data import Dataset
from transformers import AutoTokenizer, LlamaTokenizerFast, LlamaModel, BertModel
from sentence_transformers import SentenceTransformer
from ast import literal_eval
import pandas as pd

class Datasetor(Dataset):
    def __init__(self, csv,
                 model_name,
                 sentence_bert_name,
                 win_len,
                 step):
        self.csv = csv
        self.sentence_bert = SentenceTransformer(sentence_bert_name)
        if 'llama' in model_name:
            self.tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
            self.model = LlamaModel.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)

        self.sentence, self.label_group = self.bulk_data(win_len=win_len, step_len=step)

    def flat_data(self):
        sentence = []
        label_group = []
        for index in range(len(self.csv)):
            sentence += literal_eval(self.csv['sentence'][index])
            label_group += literal_eval(self.csv['label_group'][index])

        return sentence, label_group

    def bulk_data(self, win_len, step_len):
        sentence, label_group = self.flat_data()
        result_sentence = []
        result_label_group = []
        for i in range(0, len(sentence) - win_len + 1, step_len):
            result_sentence.append(sentence[i:i + win_len])
            result_label_group.append(label_group[i:i + win_len])

        return result_sentence, result_label_group

    def __len__(self):
        return len(self.sentence)

    def get_tokens(self):
        pass

    def get_labels(self):
        pass

    def get_bbox(self):
        pass

    def __getitem__(self, item):
        pass



if __name__ == "__main__":
    csv = pd.read_csv('../data/train_choi.csv')
    obj = Datasetor(csv=csv,
                    model_name='bert-base-uncased',
                    sentence_bert_name='sentence-transformers/all-MiniLM-L6-v2')





