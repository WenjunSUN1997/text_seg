import torch

class TokenEncoder(torch.nn.Module):

    def __init__(self, sim_dim):
        super(TokenEncoder, self).__init__()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=sim_dim,
                                                              nhead=8,
                                                              batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer,
                                                   num_layers=2)
        self.normalize = torch.nn.LayerNorm(normalized_shape=sim_dim)

    def forward(self, bert_feature):
        batch_num = bert_feature.shape[0]
        result = []
        for batch_index in range(batch_num):
            output_encoder = self.encoder(bert_feature[batch_index])
            result.append(output_encoder)

        result = torch.mean(torch.stack(result), dim=-2)
        return result