import torch
from model_config.token_encoder import TokenEncoder
from model_config.sentence_encoder import SentenceEncoder
from model_config.pos_embedding import PosEmbeder
from model_config.conv import FigConv
from model_config.partial_encoder import PartialEncoder
from model_config.deconv import DeConv
from model_config.flatten import flatten_fig
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
        pass
