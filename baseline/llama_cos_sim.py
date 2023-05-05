import torch
from baseline.bert_cos_sim import BertCosSim

class LlamaCosSim(BertCosSim):
    def __init__(self, bert_model, threshold, feature_type):
        super(LlamaCosSim, self).__init__(bert_model, threshold)
        self.feature_type = feature_type

    def judge_sim(self, input_ids, attention_mask):
        bert_feature = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 output_hidden_states=True,
                                 return_dict=True
                                 )['last_hidden_state']
        if self.feature_type == 'max':
            llama_feature, _ = torch.max(bert_feature, dim=1)
        elif self.feature_type == 'mean':
            llama_feature, _ = torch.mean(bert_feature, dim=1)

        cos_sim = torch.cosine_similarity(llama_feature[0], llama_feature[1], dim=0)
        if cos_sim <= self.threshold:
            return 1
        else:
            return 0