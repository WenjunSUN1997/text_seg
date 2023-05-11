import torch
from model_config.flatten import flatten_fig

class PartialEncoder(torch.nn.Module):
    def __init__(self):
        super(PartialEncoder, self).__init__()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=6,
                                                              nhead=2,
                                                              batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer,
                                                   num_layers=2)

    def forward(self, fig):
        batch_size, sim_dim, m, n = fig.shape
        input_encoder = flatten_fig(fig)
        output_encoder = self.encoder(input_encoder)
        result = output_encoder.view(batch_size, sim_dim, m, n)
        return result
