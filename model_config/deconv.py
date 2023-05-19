import torch

class DeConv(torch.nn.Module):
    def __init__(self):
        super(DeConv, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(in_channels=6, out_channels=12,
                                                 kernel_size=(2, 2))
        self.unpooling = torch.nn.MaxUnpool2d(kernel_size=(2, 2))
        self.activate = torch.nn.Tanh()

    def forward(self, input):
        output_deconv = self.deconv(input)
        return output_deconv
