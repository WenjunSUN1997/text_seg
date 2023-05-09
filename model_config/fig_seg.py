import torch
from model_config.token_encoder import TokenEncoder
from model_config.sentence_encoder import SentenceEncoder
from model_config.pos_embedding import PosEmbeder

class FigSeg(torch.nn.Module):
    def __init__(self, token_encoder_flag,
                 sentence_encoder_flag,
                 partial_encoder_flag,
                 sim_dim,
                 bert_model,
                 llama_flag,
                 bbox_flag):
        super(FigSeg, self).__init__()
        self.token_encoder_flag = token_encoder_flag
        self.sentence_encoder_flag = sentence_encoder_flag
        self.partial_encoder_flag = partial_encoder_flag
        self.llama_flag = llama_flag
        self.bbox_flag = bbox_flag
        self.sim_dim = sim_dim
        self.bert_model = bert_model
        self.token_encoder = TokenEncoder(sim_dim=sim_dim)
        if llama_flag:
            self.sentence_encoder = SentenceEncoder(sim_dim=sim_dim + 768)
            self.pos_embeder = PosEmbeder(sim_dim=sim_dim + 768)
        else:
            self.sentence_encoder = SentenceEncoder(sim_dim=2 * sim_dim)
            self.pos_embeder = PosEmbeder(sim_dim=2 * sim_dim)

    def get_bert_feature(self, data):
        batch_num = data['input_ids'].shape[0]
        bert_feature_result = []
        for batch_index in range(batch_num):
            feature = self.bert_model(input_ids=data['input_ids'][batch_index],
                                      attention_mask=data['attention_mask'][batch_index])
            feature = feature['last_hidden_state']
            bert_feature_result.append(feature)

        return torch.stack(bert_feature_result)

    def forward(self, data):
        bert_feature = self.get_bert_feature(data)
        if self.token_encoder_flag:
            token_feature = self.token_encoder(bert_feature)
        else:
            token_feature = torch.mean(bert_feature, dim=2)

        sentence_feature = torch.cat([token_feature, data['sentence_bert_vec']],
                                     dim=2)
        if self.sentence_encoder_flag:
            sentence_feature = self.sentence_encoder(sentence_feature)

        if self.bbox_flag:
            pos_embedding = self.pos_embeder(data)









        return data