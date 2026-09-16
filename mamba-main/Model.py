import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba  # Cần đảm bảo đã cài Mamba

class DOAMambaNet(nn.Module):
    def __init__(self, num_antennas=10, num_snapshots=100, d_model=64, mamba_layers=3, num_classes=181):
        super().__init__()
        self.num_antennas = num_antennas
        self.num_snapshots = num_snapshots
        self.input_dim = 2 * num_antennas  # real + imag

        # Mamba block
        self.mamba_stack = nn.Sequential(*[
        Mamba(d_model=d_model) for _ in range(mamba_layers)])
        self.pre_norm = nn.LayerNorm(d_model)
        self.input_proj = nn.Linear(self.input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, self.input_dim)

        # CNN layers sau khi tạo ma trận tương quan Hermitian
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):  # x: (B, 2, M, T)
        B, C, M, T = x.shape

        # Kiểm tra và reshape dữ liệu nếu có shape đặc biệt từ Mamba
        if x.ndim == 4 and C == 2:
            x = x.permute(0, 3, 1, 2).reshape(B, T, -1)  # (B, T, 2M)

        x = self.input_proj(x)        # (B, T, d_model)
        x = self.pre_norm(x)          # <--- chuẩn hoá trước khi vào Mamba
        x = self.mamba_stack(x)       # (B, T, d_model)
        x = torch.clamp(x, min=-10, max=10)
        x = self.norm(x)
        x = self.output_proj(x)       # (B, T, 2M)

        # reconstruct signal (B, T, 2M) -> (B, 2, M, T)
        x = x.reshape(B, T, C, M).permute(0, 2, 3, 1)  # (B, 2, M, T)
        x = x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-6)   

        # Tạo ma trận tương quan Hermitian: (B, 2, M, T) -> (B, 2, M, M)
        real = x[:, 0]  # (B, M, T)
        imag = x[:, 1]
        complex_x = torch.complex(real, imag)  # (B, M, T)

        conj_x = torch.conj(complex_x)
        hermitian = torch.matmul(complex_x, conj_x.transpose(1, 2)) / T  # (B, M, M)

        # tách real và imag -> (B, 2, M, M)
        out = torch.stack([hermitian.real, hermitian.imag], dim=1)

        # CNN xử lý
        features = self.cnn_layers(out)
        logits = self.classifier(features)

        return logits
