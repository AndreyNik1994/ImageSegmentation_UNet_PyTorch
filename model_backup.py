import torch.nn as nn
import os
import torch
import torch.nn.functional as F
from collections import namedtuple


BaseModelOutput = namedtuple('BaseModelOutput', ['last_hidden_state'])


class UNet(nn.Module):
    def __init__(self, in_channels, out_classes, encoder_channels):
        super().__init__()

        ### Encoder
        self.encoder = nn.ModuleList()
        self.out_classes = out_classes
        self.img_width = encoder_channels[-1]
        self.img_height = encoder_channels[-1]
        self.afunc = 'relu'

        current_in = in_channels

        for ch in encoder_channels[:-1]:
            self.encoder.append(self._conv_block(current_in, ch))
            current_in = ch

        ### Самый глубокий слой
        self.bottleneck = self._conv_block(encoder_channels[-2], encoder_channels[-1])

        ### Decoder
        self.decoder = nn.ModuleList()
        current_in = encoder_channels[-1]

        for ch in reversed(encoder_channels[:-1]):
            self.decoder.append(self._conv_trans_block(current_in, ch))
            current_in = ch

        ### Финальный слой
        self.final = nn.Conv2d(encoder_channels[0], out_classes, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))

    def _conv_trans_block(self, in_channels, out_channels):
        return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                             self._conv_block(out_channels * 2, out_channels))

    def save(self, name: str, destination: str | os.PathLike):
        save_path = os.path.join(destination)
        os.makedirs(save_path, exist_ok=True)

        name += "_detection.pth"
        save_path = os.path.join(save_path, name)
        torch.save(self.state_dict(), save_path)

    def forward(self, x):
        skip_connections = []

        for block in self.encoder:
            x = block(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, 2)

        x = self.bottleneck(x)
        for i, block in enumerate(self.decoder):
            x = block[0](x)
            skip = skip_connections[-(i + 1)]
            x = torch.cat((skip, x), dim=1)
            x = block[1](x)

        x = self.final(x)
        return x
