import torch

class FigConv(torch.nn.Module):
    def __init__(self, num_in_channel, num_out_channel, sim_dim, kernel_size):
        super(FigConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=num_in_channel,
                                    out_channels=num_out_channel,
                                    kernel_size=(kernel_size, sim_dim),
                                    stride=1,
                                    padding=0)
        self.pooling = torch.nn.MaxPool2d(kernel_size=(2, 2), return_indices=True)
        self.activate = torch.nn.Tanh()

    def forward(self, fig):
        output = self.conv(self.activate(fig))
        return output
