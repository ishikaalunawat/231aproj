import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import argparse

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
    def __init__(self, args):
        super().__init__()
        self.patch_size = args.patch_size
        self.n_channels = args.n_channels
        self.latent_size = args.latent_size

        self.linear_projection = nn.Linear(self.patch_size * self.patch_size * self.n_channels, self.latent_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.latent_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, (args.img_size // self.patch_size) ** 2 + 1, self.latent_size))
        
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

class VisionTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = InputEmbedding(args)
        self.encoder = nn.Sequential(
            *[EncoderBlock(args.latent_size, args.num_heads, 4, args.dropout) for _ in range(args.num_encoders)]
        )
        self.conv_head = nn.Sequential(
            nn.Conv2d(args.latent_size, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 3, kernel_size=1)
        )

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = self.embedding(x)
        x = self.encoder(x)
        
        # Remove cls token and reshape
        x = x[:, 1:, :]
        h_patches, w_patches = height // self.embedding.patch_size, width // self.embedding.patch_size
        x = x.permute(0, 2, 1).contiguous().view(batch_size, -1, h_patches, w_patches)
        
        # Upsample back to the original image size
        x = nn.functional.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)
        x = self.conv_head(x)
        
        # Reshape to (B, N, 3)
        x = x.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 3)
        return x

def main():
    parser = argparse.ArgumentParser(description='Vision Transformer in PyTorch')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--patch-size', type=int, default=16, help='patch size for images (default : 16)')
    parser.add_argument('--latent-size', type=int, default=256, help='latent size (default : 256)')
    parser.add_argument('--n-channels', type=int, default=3, help='number of channels in images (default : 3 for RGB)')
    parser.add_argument('--num-heads', type=int, default=12, help='(default : 12)')
    parser.add_argument('--num-encoders', type=int, default=12, help='number of encoders (default : 12)')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout value (default : 0.1)')
    parser.add_argument('--img-size', type=int, default=224, help='image size to be reshaped to (default : 224)')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size (default : 4)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs (default : 10)')
    parser.add_argument('--lr', type=float, default=1e-2, help='base learning rate (default : 0.01)')
    parser.add_argument('--weight-decay', type=float, default=3e-2, help='weight decay value (default : 0.03)')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor()
    ])
    
    train_data = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
    valid_data = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)

    model = VisionTransformer(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()  # Change this based on your specific task requirements

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # Placeholder for 3D points labels; replace with actual labels
            labels = torch.randn_like(outputs).to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, _ in valid_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                labels = torch.randn_like(outputs).to(device)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        print(f'Validation Loss: {total_loss / len(valid_loader)}')

if __name__ == "__main__":
    # main()