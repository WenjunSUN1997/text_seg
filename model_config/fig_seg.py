import torch
from model_config.token_encoder import TokenEncoder
from model_config.sentence_encoder import SentenceEncoder
from model_config.pos_embedding import PosEmbeder
from model_config.conv import FigConv
from model_config.partial_encoder import PartialEncoder
from model_config.deconv import DeConv
from model_config.flatten import flatten_fig
import math

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
        self.conv = FigConv()
        self.partial_encoder = PartialEncoder()
        self.deconv = DeConv()
        self.linear = torch.nn.Linear(in_features=12, out_features=2)
        self.activate = torch.nn.Tanh()
        self.sentence_encoder = SentenceEncoder(sim_dim=sim_dim)
        self.pos_embeder = PosEmbeder(sim_dim=sim_dim)

    def get_bert_feature(self, data):
        batch_num = data['input_ids'].shape[0]
        bert_feature_result = []
        for batch_index in range(batch_num):
            feature = self.bert_model(input_ids=data['input_ids'][batch_index],
                                      attention_mask=data['attention_mask'][batch_index])
            feature = feature['last_hidden_state']
            bert_feature_result.append(feature)

        return torch.stack(bert_feature_result)

    def get_sim(self, feature):
        batch_size = feature.shape[0]
        result = []
        for batch_index in range(batch_size):
            similarity_matrix = torch.cosine_similarity(feature[batch_index].unsqueeze(1),
                                                        feature[batch_index].unsqueeze(0),
                                                        dim=-1)
            result.append(similarity_matrix)

        return torch.stack(result)

    def forward(self, data):
        bert_feature = self.get_bert_feature(data)
        if self.token_encoder_flag:
            token_feature = self.token_encoder(bert_feature)
        else:
            token_feature = torch.mean(bert_feature, dim=2)

        # sentence_feature = torch.cat([token_feature, data['sentence_bert_vec']],
        #                              dim=2)
        sentence_feature = token_feature
        if self.sentence_encoder_flag:
            sentence_feature = self.sentence_encoder(sentence_feature)

        if self.bbox_flag:
            pos_embedding = self.pos_embeder(data)
            sentence_feature = sentence_feature + pos_embedding

        sim_token_feature = self.get_sim(token_feature)
        sim_sentence_bert = self.get_sim(data['sentence_bert_vec'])
        sim_sentence_feature = self.get_sim(sentence_feature)
        fig = torch.cat([sim_token_feature.unsqueeze(1),
                         sim_sentence_bert.unsqueeze(1),
                         sim_sentence_feature.unsqueeze(1)], dim=1)
        conv_feature = self.conv(fig)
        if self.partial_encoder_flag:
            conv_feature = self.partial_encoder(conv_feature)

        deconv_feature = self.deconv(conv_feature)
        input_linear = self.activate(flatten_fig(deconv_feature))
        output_linear = self.linear(input_linear)
        prob_fig = torch.softmax(output_linear, dim=-1)
        seg_result, seg_prob = self.post_process(prob_fig)
        result = {'token_sim': sim_token_feature,
                  'sentence_sim': sim_sentence_feature,
                  'prob_fig': prob_fig,
                  'seg_result': seg_result,
                  'seg_prob': seg_prob,
                  }

        return result

    def post_process(self, prob_fig):
        batch_size = prob_fig.shape[0]
        sentence_num = int(math.sqrt(prob_fig.shape[1]))
        max_index = torch.argmax(prob_fig, dim=-1).view(batch_size,
                                                        sentence_num, sentence_num)
        result = []
        seg_prob = []
        for batch_index in range(batch_size):
            for first_sen in range(sentence_num-1):
                seg_prob.append(prob_fig[batch_index][1 + first_sen * (sentence_num + 1)])
                if max_index[batch_index][first_sen][first_sen+1] == 0:
                    result.append(0)
                else:
                    result.append(1)

        return result, torch.stack(seg_prob)

