import torch
import torch.nn as nn
import torch.nn.functional as F

# 张量调整
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

# 注意力机制
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

# 方差通道
class VarianceAugmentation(nn.Module):
    """Add variance as an additional feature channel"""

    def forward(self, x):
        # x shape: [batch, 1, seq_len]
        variance = torch.var(x, dim=2, keepdim=True)  # [batch, 1, 1]
        normalized_var = (variance - variance.mean()) / (variance.std() + 1e-8)

        # 将方差重复到与输入相同的长度
        normalized_var = normalized_var.repeat(1, 1, x.size(2))  # [batch, 1, seq_len]

        return torch.cat([x, normalized_var], dim=1)  # [batch, 2, seq_len]

# 低通滤波通道
class LowPassAugmentation(nn.Module):
    """Add low-pass filtered signal as an additional feature channel"""

    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        # Create a simple low-pass filter kernel (moving average)
        self.register_buffer('kernel', torch.ones(1, 1, kernel_size) / kernel_size)
        self.pad = kernel_size // 2

    def forward(self, x):
        # x shape: [batch, 1, seq_len]
        # Apply low-pass filtering
        low_pass = F.conv1d(
            x,
            self.kernel,
            padding=self.pad,
            stride=1
        )

        # Normalize the low-pass signal
        normalized_low_pass = (low_pass - low_pass.mean(dim=2, keepdim=True)) / \
                              (low_pass.std(dim=2, keepdim=True) + 1e-8)

        return torch.cat([x, normalized_low_pass], dim=1)  # [batch, 2, seq_len]

# BiGRU模型
class BiGRUBlock(nn.Module):
    """Bidirectional GRU layer with feature fusion"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)
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

# Lora调试
class LoRALayer(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=16):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha

        # Freeze original parameters
        for param in base_layer.parameters():
            param.requires_grad = False

        # Add LoRA parameters
        if isinstance(base_layer, nn.Linear):
            in_dim = base_layer.in_features
            out_dim = base_layer.out_features
            self.lora_A = nn.Parameter(torch.randn(in_dim, rank))
            self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        elif isinstance(base_layer, nn.Conv1d):
            in_ch = base_layer.in_channels
            out_ch = base_layer.out_channels
            kernel_size = base_layer.kernel_size[0]
            self.lora_A = nn.Parameter(torch.randn(in_ch * kernel_size, rank))
            self.lora_B = nn.Parameter(torch.zeros(rank, out_ch * kernel_size))
        else:
            raise ValueError("Unsupported layer type for LoRA")

        self.scaling = alpha / rank

    def forward(self, x):
        base_output = self.base_layer(x)

        if isinstance(self.base_layer, nn.Linear):
            lora_weights = torch.matmul(self.lora_A, self.lora_B).T
            lora_output = F.linear(x, lora_weights) * self.scaling
        elif isinstance(self.base_layer, nn.Conv1d):
            lora_weights = torch.matmul(self.lora_A, self.lora_B).view(
                self.base_layer.out_channels,
                self.base_layer.in_channels,
                self.base_layer.kernel_size[0]
            )
            lora_output = F.conv1d(x, lora_weights, padding=self.base_layer.padding[0]) * self.scaling

        return base_output + lora_output

# VAE降噪模型
class DenoisingVAE(nn.Module):
    def __init__(self, latent_dim=128, feature_length=1024):
        super().__init__()
        self.feature_length = feature_length

        # Encoder
        self.encoder = nn.Sequential(
            LowPassAugmentation(),  # [batch, 2, seq_len]

            # First downsampling block
            nn.Conv1d(2, 32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            BiGRUBlock(32, 16),
            nn.MaxPool1d(2),  # 512

            # Second block
            ResidualBlock(32),
            BiGRUBlock(32, 16),
            ResidualBlock(32),

            # Third downsampling block
            nn.Conv1d(32, 64, 3, stride=1, padding=1),  # 512
            nn.BatchNorm1d(64),
            nn.GELU(),
            BiGRUBlock(64, 32),
            nn.MaxPool1d(2),  # 256

            # Fourth block
            ResidualBlock(64),
            BiGRUBlock(64, 32),
            ResidualBlock(64),

            # Fifth downsampling block
            nn.Conv1d(64, 128, 3, stride=1, padding=1),  # 256
            nn.BatchNorm1d(128),
            nn.GELU(),
            BiGRUBlock(128, 64),
            nn.MaxPool1d(2),  # 128

            # Final block
            ResidualBlock(128),
            BiGRUBlock(128, 64),
            ResidualBlock(128),

            nn.Flatten()  # 128 * (feature_length//8)
        )

        self.encoder_output_dim = 128 * (feature_length // 8)
        self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_var = nn.Linear(self.encoder_output_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.encoder_output_dim),
            View((-1, 128, feature_length // 8)),  # 128 channels, 128 length

            ResidualBlock(128),
            BiGRUBlock(128, 64),
            ResidualBlock(128),

            nn.ConvTranspose1d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 256
            nn.BatchNorm1d(64),
            nn.GELU(),
            BiGRUBlock(64, 32),

            ResidualBlock(64),
            BiGRUBlock(64, 32),
            ResidualBlock(64),

            nn.ConvTranspose1d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 512
            nn.BatchNorm1d(32),
            nn.GELU(),
            BiGRUBlock(32, 16),

            ResidualBlock(32),
            BiGRUBlock(32, 16),
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



# 测试
if __name__ == "__main__":
    model = DenoisingVAE(feature_length=1024)
    x = torch.randn(32, 1, 1024)
    reconstruction, mu, logvar = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")