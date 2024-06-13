import torch
import torch.nn as nn

class PatchExtractor(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch_size, -1, channels * self.patch_size * self.patch_size)
        return patches

class InputEmbedding(nn.Module):
    def __init__(self, patch_size, n_channels, latent_size, img_height=368, img_width=512):
        super().__init__()
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.latent_size = latent_size

        self.linear_projection = nn.Linear(self.patch_size * self.patch_size * self.n_channels, self.latent_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.latent_size))
        num_patches = (img_height // self.patch_size) * (img_width // self.patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.latent_size))

    def forward(self, x):
        patches = PatchExtractor(self.patch_size)(x)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, self.linear_projection(patches)), dim=1)
        x += self.pos_embedding
        return x

class EncoderBlock(nn.Module):
    def __init__(self, latent_size, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(latent_size)
        self.attn = nn.MultiheadAttention(latent_size, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(latent_size)
        self.mlp = nn.Sequential(
            nn.Linear(latent_size, latent_size * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_size * mlp_ratio, latent_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, latent_size, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(latent_size)
        self.attn = nn.MultiheadAttention(latent_size, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(latent_size)
        self.mlp = nn.Sequential(
            nn.Linear(latent_size, latent_size * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_size * mlp_ratio, latent_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class StudentModel(nn.Module):
    def __init__(self, patch_size, n_channels, latent_size, num_heads, num_encoders, num_decoders, dropout, img_height=368, img_width=512):
        super().__init__()
        self.embedding = InputEmbedding(patch_size, n_channels, latent_size, img_height, img_width)
        self.encoder = nn.Sequential(
            *[EncoderBlock(latent_size, num_heads, 4, dropout) for _ in range(num_encoders)]
        )
        self.decoder = nn.Sequential(
            *[DecoderBlock(latent_size, num_heads, 4, dropout) for _ in range(num_decoders)]
        )
        self.conv_head = nn.Sequential(
            nn.Conv2d(latent_size, 512, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 3, kernel_size=1)
        )
        self.patch_size = patch_size
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.00025)
        self.optimizer.zero_grad()
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.decoder(x)

        # Remove cls token and reshape
        x = x[:, 1:, :]
        h_patches, w_patches = height // self.patch_size, width // self.patch_size
        x = x.permute(0, 2, 1).contiguous().view(batch_size, -1, h_patches, w_patches)

        # Upsample back to the original image size
        x = nn.functional.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)
        x = self.conv_head(x)

        # Reshape to (B, N, 3)
        x = x.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 3)
        return x

    def learn(self, x, y):
        y_pred = self.forward(x)
        self.optimizer.zero_grad()
        l = self.loss(y_pred, y)
        loss_val = l.item()
        l.backward(retain_graph=True)
        self.optimizer.step()
        return loss_val