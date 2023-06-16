import torch

class TwoLevelTrans(torch.nn.Module):
    def __init__(self, sim_dim):
        super(TwoLevelTrans, self).__init__()
        self.sim_dim = sim_dim
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=sim_dim,
                                                              nhead=8,
                                                              batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer,
                                                   num_layers=2)
        self.linear = torch.nn.Linear(in_features=2 * sim_dim, out_features=2)

    def judge_sim(self, sentence_bert_vec):
        output_encoder = self.encoder(sentence_bert_vec)
        prob_log = self.linear(output_encoder.view(2 * self.sim_dim))
        prob = torch.softmax(prob_log, dim=-1)
        label = torch.argmax(prob)
        return {'prob': prob,
                'label': label.item()}

    def judge_sim_new(self, sentence_bert_vec):
        prob_list = []
        label_list = []
        output_encoder = self.encoder(sentence_bert_vec)
        for index in range(output_encoder.shape[0] - 1):
            prob_log = self.linear(torch.cat((output_encoder[index],
                                              output_encoder[index + 1])))
            prob = torch.softmax(prob_log, dim=-1)
            prob_list.append(prob)
            label_list.append(torch.argmax(prob).item())

        return {'prob': prob_list,
                'label': label_list}

    def forward(self, data):
        seg_result = []
        prob = []
        batch_num = data['sentence_bert_vec'].shape[0]
        for batch_index in range(batch_num):
            output_judge_layer = self.judge_sim_new(data['sentence_bert_vec'][batch_index])
            prob += output_judge_layer['prob']
            seg_result += output_judge_layer['label']

        return {'seg_result': seg_result,
                'prob': torch.stack(prob)}