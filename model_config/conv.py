import torch

class FigConv(torch.nn.Module):
    def __init__(self):
        super(FigConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=6,
                                    kernel_size=(2, 2))
        self.pooling = torch.nn.MaxPool2d(kernel_size=(2, 2), return_indices=True)
        self.activate = torch.nn.Tanh()

    def forward(self, fig):
        output = self.conv(self.activate(fig))
        return output
