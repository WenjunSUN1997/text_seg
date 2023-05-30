import torch
from model_config.flatten import flatten_fig

class PartialEncoder(torch.nn.Module):
    def __init__(self, d_model):
        super(PartialEncoder, self).__init__()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model,
                                                              nhead=4,
                                                              batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer,
                                                   num_layers=2)

    def forward(self, fig):
        if len(fig.shape) == 4:
            result = self.forward_fig(fig)
        else:
            result = self.forward_encoder(fig)
        return result

    def forward_fig(self, fig):
        batch_size, sim_dim, m, n = fig.shape
        input_encoder = flatten_fig(fig)
        output_encoder = self.encoder(input_encoder)
        result = output_encoder.view(batch_size, sim_dim, m, n)
        return result

    def forward_encoder(self, fig):
        output_encoder = self.encoder(fig)
        return output_encoder
