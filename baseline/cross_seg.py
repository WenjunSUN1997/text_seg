import torch

class CrossSeg(torch.nn.Module):

    def __init__(self, bert_model, sim_dim, feature_type):
        super(CrossSeg, self).__init__()
        self.bert_model = bert_model
        self.sim_dim = sim_dim
        self.feature_type = feature_type
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=sim_dim,
                                                              nhead=8,
                                                              batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer,
                                                   num_layers=2)
        self.linear = torch.nn.Linear(in_features=sim_dim, out_features=2)

    def judge_sim(self, input_ids, attention_mask):
        bert_feature = self.bert_model(input_ids=input_ids,
                                       attention_mask=attention_mask)['last_hidden_state']
        bert_feature = bert_feature.view(-1, self.sim_dim)
        output_encoder = self.encoder(bert_feature)
        if self.feature_type == 'max':
            input_linear, _ = torch.max(output_encoder, dim=0)
        else:
            input_linear = torch.mean(output_encoder, dim=0)
        prob_log = self.linear(input_linear)
        prob = torch.softmax(prob_log, dim=-1)
        label = torch.argmax(prob)
        return {'prob': prob,
                'label': label.item()}

    def forward(self, data):
        seg_result = []
        prob = []
        batch_num = data['input_ids'].shape[0]
        for batch_index in range(batch_num):
            input_ids = data['input_ids'][batch_index]
            attention_mask = data['attention_mask'][batch_index]
            output_judge_layer = self.judge_sim(input_ids, attention_mask)
            prob.append(output_judge_layer['prob'])
            seg_result.append(output_judge_layer['label'])

        return {'seg_result': seg_result,
                'prob': torch.stack(prob)}
