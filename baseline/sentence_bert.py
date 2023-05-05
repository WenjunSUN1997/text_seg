import torch

class SentenceBertCosSim(torch.nn.Module):
    def __init__(self, threshold):
        super(SentenceBertCosSim, self).__init__()
        self.threshold = threshold

    def judge_sim(self, sentence_bert):
        cos_sim = torch.cosine_similarity(sentence_bert[0], sentence_bert[1], dim=0)
        if cos_sim <= self.threshold:
            return 1
        else:
            return 0

    def forward(self, data):
        seg_result = []
        batch_num = data['sentence_bert_vec'].shape[0]
        for batch_index in range(batch_num):
            seg_result.append(self.judge_sim(data['sentence_bert_vec'][batch_index]))

        return {'seg_result': seg_result}

