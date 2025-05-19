import torch
import torch.nn as nn


class Conv1dBN(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride = 1, padding = 1, bias = False):
        super(Conv1dBN, self).__init__()
        self.conv = nn.Conv1d(in_chans, out_chans, kernel_size, stride = stride, padding = padding, bias = bias)
        self.bn = nn.BatchNorm1d(out_chans)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride = 1, with_conv_shortcut = False):
        super(ConvBlock, self).__init__()
        self.conv1 = Conv1dBN(in_chans, out_chans, kernel_size, stride = stride, padding = kernel_size // 2)
        self.conv2 = Conv1dBN(out_chans, out_chans, kernel_size, padding = kernel_size // 2)
        self.with_conv_shortcut = with_conv_shortcut
        if self.with_conv_shortcut:
            self.shortcut = Conv1dBN(in_chans, out_chans, kernel_size, stride = stride, padding = kernel_size // 2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.with_conv_shortcut:
            residual = self.shortcut(x)
        out = out + residual
        return out


# BiGRU模型
class BiGRUBlock(nn.Module):
    """Bidirectional GRU layer with feature fusion"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, bidirectional = True, batch_first = True)
        self.projection = nn.Linear(2 * hidden_dim, input_dim)

    def forward(self, x):
        # x shape: [batch, channels, seq_len]
        batch, channels, seq_len = x.size()

        # Reshape for GRU: [batch, seq_len, channels]
        x = x.permute(0, 2, 1)

        # Process through BiGRU
        gru_out, _ = self.gru(x)  # [batch, seq_len, 2*hidden_dim]

        # Project back to original dimension
        projected = self.projection(gru_out)  # [batch, seq_len, channels]

        # Reshape back: [batch, channels, seq_len]
        return projected.permute(0, 2, 1)


class FreqNet(nn.Module):
    def __init__(self, in_chans = 1, base_channels = 64):
        super(FreqNet, self).__init__()

        # Encoder
        self.encoder1 = Conv1dBN(in_chans, base_channels, 15, stride = 1, padding = 7)
        self.encoder2 = ConvBlock(base_channels, base_channels, 15)

        self.down1 = ConvBlock(base_channels, base_channels * 2, 15, stride = 2, with_conv_shortcut = True)
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 2, 15)

        self.down2 = ConvBlock(base_channels * 2, base_channels * 4, 15, stride = 2, with_conv_shortcut = True)
        self.encoder4 = ConvBlock(base_channels * 4, base_channels * 4, 15)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4, 15)

        # Decoder
        self.decoder1 = ConvBlock(base_channels * 4, base_channels * 4, 15)
        self.fusion_gru1 = BiGRUBlock(base_channels * 4, base_channels * 4)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'linear', align_corners = True),
            Conv1dBN(base_channels * 4, base_channels * 2, 15, padding = 7)
        )

        self.decoder2 = ConvBlock(base_channels * 2, base_channels * 2, 15)
        self.fusion_gru2 = BiGRUBlock(base_channels * 2, base_channels * 2)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'linear', align_corners = True),
            Conv1dBN(base_channels * 2, base_channels, 15, padding = 7)
        )

        self.decoder3 = ConvBlock(base_channels, base_channels, 15)

        # Final output
        self.final_conv = nn.Conv1d(base_channels, in_chans, 1)

    def forward(self, x):
        # Encoder path
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)

        e3 = self.down1(e2)
        e4 = self.encoder3(e3)

        e5 = self.down2(e4)
        e6 = self.encoder4(e5)

        # Bottleneck
        b = self.bottleneck(e6)

        # Decoder path with skip connections
        d1 = self.decoder1(b)
        d1 = self.fusion_gru1(d1) + e6
        d2 = self.up1(d1) + e4

        d3 = self.decoder2(d2)
        d3 = self.fusion_gru2(d3) + e3
        d4 = self.up2(d3) + e1

        d5 = self.decoder3(d4)

        # Final output
        out = self.final_conv(d5)

        return out


if __name__ == '__main__':
    # Test the network
    batch_size = 8
    freq_points = 2048  # Number of frequency sampling points
    x = torch.randn(batch_size, 1, freq_points)
    model = FreqNet(in_chans = 1)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
