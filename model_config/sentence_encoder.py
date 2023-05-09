import torch

class SentenceEncoder(torch.nn.Module):

    def __init__(self, sim_dim):
        super(SentenceEncoder, self).__init__()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=sim_dim,
                                                              nhead=8,
                                                              batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer,
                                                   num_layers=2)
        self.normalize = torch.nn.LayerNorm(normalized_shape=sim_dim)

    def forward(self, bert_feature):
        output_encoder = self.encoder(bert_feature)
        return output_encoder
