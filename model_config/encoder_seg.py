import torch
from model_config.token_encoder import TokenEncoder
from model_config.sentence_encoder import SentenceEncoder
from model_config.pos_embedding import PosEmbeder
from model_config.conv import FigConv
from model_config.partial_encoder import PartialEncoder
from model_config.deconv import DeConv
from model_config.flatten import flatten_fig, flatten_encoder
import math
from model_config.fig_seg import FigSeg

class EncoderSeg(FigSeg):
    def __init__(self, token_encoder_flag,
                 sentence_encoder_flag,
                 partial_encoder_flag,
                 sim_dim,
                 bert_model,
                 llama_flag,
                 bbox_flag):
        super(EncoderSeg, self).__init__(token_encoder_flag,
                                         sentence_encoder_flag,
                                         partial_encoder_flag,
                                         sim_dim,
                                         bert_model,
                                         llama_flag,
                                         bbox_flag)

    def forward(self, data):
        bert_feature = self.get_bert_feature(data)
        sentence_bert_feature = data['sentence_bert_vec']
        if self.token_encoder_flag:
            token_feature = self.token_encoder(bert_feature)
        else:
            token_feature = torch.mean(bert_feature, dim=2)

        sentence_feature = token_feature
        if self.sentence_encoder_flag:
            sentence_encoder_feature = self.sentence_encoder(sentence_feature)
        else:
            sentence_encoder_feature = torch.mean(bert_feature, dim=2)

        if self.bbox_flag:
            pos_embedding = self.pos_embeder(data)
            fig_feature = torch.stack((token_feature,
                                       sentence_encoder_feature,
                                       sentence_bert_feature,
                                       pos_embedding),
                                      dim=1)
        else:
            fig_feature = torch.stack((token_feature,
                                       sentence_encoder_feature,
                                       sentence_bert_feature),
                                      dim=1)

        output_conv_2 = self.conv_2(fig_feature).squeeze(-1)
        output_conv_4 = self.conv_4(fig_feature)
        output_conv_6 = self.conv_6(fig_feature)
        if self.partial_encoder_flag:
            fig_feature_final = self.partial_encoder(flatten_encoder(output_conv_2))
        else:
            fig_feature_final = output_conv_2

        input_linear = self.activate(fig_feature_final)
        output_linear = self.linear(input_linear)
        prob = torch.softmax(output_linear, dim=-1)
        label = torch.argmax(prob, dim=-1)
        result = {'prob': prob,
                  'seg_result': label}
        return result

