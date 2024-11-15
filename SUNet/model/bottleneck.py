import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling over H and W
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)  # Shape: (B, C)
        y = self.fc(y).view(b, c, 1, 1)  # Shape: (B, C, 1, 1)
        return x * y  # Channel-wise multiplication

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2  # Ensure the output size matches the input size
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (B, C, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        x = torch.cat([avg_out, max_out], dim=1)  # Shape: (B, 2, H, W)
        x = self.conv(x)  # Shape: (B, 1, H, W)
        return self.sigmoid(x)  # Spatial attention map

class Bottleneck(nn.Module):
    def __init__(self, channels, block, reduction=16):
        super(Bottleneck, self).__init__()

        self.block = block
        self.se_block = SEBlock(channels, reduction)
        self.spatial_attn = SpatialAttention()
    
    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.size()
        H, W = self.block.input_resolution 
        assert H * W == L, "Input feature length L must be H x W"
        x = self.block(x)

        # Reshape to (B, C, H, W)
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        
        # Apply Squeeze-and-Excitation block
        x = self.se_block(x)
        
        # Apply Spatial Attention block
        sa = self.spatial_attn(x)
        x = x * sa  # Element-wise multiplication with spatial attention map
        
        # Reshape back to (B, L, C)
        x = x.view(B, C, -1).permute(0, 2, 1).contiguous()
        return x
