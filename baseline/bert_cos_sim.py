import torch
class BertCosSim(torch.nn.Module):
    def __init__(self, bert_model, threshold):
        super(BertCosSim, self).__init__()
        self.bert_model = bert_model
        self.threshold = threshold

    def judge_sim(self, input_ids, attention_mask):
        bert_feature = self.bert_model(input_ids=input_ids,
                                       attention_mask=attention_mask)['last_hidden_state']
        cls_feature = bert_feature[:, 0, :]
        cos_sim = torch.cosine_similarity(cls_feature[0], cls_feature[1], dim=0)
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
            seg_result.append(self.judge_sim(input_ids, attention_mask))

        return_value = {'seg_result': seg_result}
        return return_value
