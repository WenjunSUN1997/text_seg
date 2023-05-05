import torch
class DoubleBert(torch.nn.Module):
    def __init__(self, bert_model, threshold):
        super(DoubleBert, self).__init__()
        self.bert_model = bert_model
        self.threshold = threshold

    def judge_sim(self, input_ids, attention_mask, sentence_bert):
        bert_feature = self.bert_model(input_ids=input_ids,
                                       attention_mask=attention_mask)['last_hidden_state']
        cls_feature = bert_feature[:, 0, :]
        max_values_0, _ = torch.max(torch.stack([cls_feature[0], sentence_bert[0]]),
                                    dim=0)
        max_values_1, _ = torch.max(torch.stack([cls_feature[1], sentence_bert[1]]),
                                    dim=0)
        cos_sim = torch.cosine_similarity(max_values_0, max_values_1, dim=0)
        if cos_sim <= self.threshold:
            return 1
        else:
            return 0

    def forward(self, data):
        batch_num = data['input_ids'].shape[0]
        seg_result = []
        for batch_index in range(batch_num):
            input_ids = data['input_ids'][batch_index]
            attention_mask = data['attention_mask'][batch_index]
            seg_result.append(self.judge_sim(input_ids,
                                             attention_mask,
                                             data['sentence_bert_vec'][batch_index]))

        return_value = {'seg_result': seg_result}
        return return_value
