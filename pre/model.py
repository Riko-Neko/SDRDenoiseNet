import torch
import torch.nn as nn

class View(nn.Module):
    """Custom tensor reshaping layer"""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution (supports transposed conv)"""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, transpose=False, padding=None, output_padding=0):
        super().__init__()
        Conv = nn.ConvTranspose1d if transpose else nn.Conv1d
        if padding is None:
            padding = kernel_size // 2

        conv_kwargs = {
            'in_channels': in_ch,
            'out_channels': in_ch,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'groups': in_ch
        }
        if transpose:
            conv_kwargs['output_padding'] = output_padding

        self.depthwise = Conv(**conv_kwargs)
        self.pointwise = Conv(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class ResidualBlock(nn.Module):
    """Residual block with channel attention"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(channels, channels, 3),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            DepthwiseSeparableConv(channels, channels, 3),
            nn.BatchNorm1d(channels)
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 8, 1),
            nn.GELU(),
            nn.Conv1d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.attention(x)
        return x + self.conv(x) * attn

class AttentionBlock(nn.Module):
    """Spatial attention mechanism"""
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, channels // 8, 1),
            nn.GELU(),
            nn.Conv1d(channels // 8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attn(x)

class DenoisingVAE(nn.Module):
    def __init__(self, latent_dim=128, feature_length=1024):
        super().__init__()
        self.feature_length = feature_length

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 3, stride=1, padding=1),  # 1024
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),  # 512
            ResidualBlock(32),
            ResidualBlock(32),
            nn.Conv1d(32, 64, 3, stride=1, padding=1),  # 512
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),  # 256
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv1d(64, 128, 3, stride=1, padding=1),  # 256
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),  # 128
            ResidualBlock(128),
            ResidualBlock(128),
            nn.Flatten()  # 128 * 128 = 16384
        )

        self.encoder_output_dim = 128 * (feature_length // 8)
        self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_var = nn.Linear(self.encoder_output_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.encoder_output_dim),
            View((-1, 128, feature_length // 8)),  # 128 channels, 128 length
            ResidualBlock(128),
            ResidualBlock(128),
            nn.ConvTranspose1d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 256
            nn.BatchNorm1d(64),
            nn.GELU(),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.ConvTranspose1d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 512
            nn.BatchNorm1d(32),
            nn.GELU(),
            ResidualBlock(32),
            ResidualBlock(32),
            nn.ConvTranspose1d(32, 1, 3, stride=2, padding=1, output_padding=1)  # 1024
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

# Test the model
if __name__ == "__main__":
    model = DenoisingVAE(feature_length=1024)
    x = torch.randn(32, 1, 1024)
    reconstruction, mu, logvar = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")