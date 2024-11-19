import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import to_2tuple, trunc_normal_
from .bottleneck import Bottleneck
from .swin_basiclayer import *
from .gc_basiclayer import * 

class CrossAttentionLayer(nn.Module):
    def __init__(self, local_dim, global_dim, num_heads=8):
        super(CrossAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.local_dim = local_dim
        self.global_dim = global_dim

        # Ensure that local_dim and global_dim are divisible by num_heads
        assert local_dim % num_heads == 0, "local_dim must be divisible by num_heads"
        assert global_dim % num_heads == 0, "global_dim must be divisible by num_heads"

        self.dim_head_local = local_dim // num_heads
        self.dim_head_global = global_dim // num_heads

        # Linear projections for local features (queries)
        self.query_proj = nn.Linear(local_dim, local_dim)

        # Linear projections for global features (keys and values)
        self.key_proj = nn.Linear(global_dim, local_dim)
        self.value_proj = nn.Linear(global_dim, local_dim)

        # Output projection
        self.out_proj = nn.Linear(local_dim, local_dim)

        # Optional: Layer normalization
        self.norm_local = nn.LayerNorm(local_dim)
        self.norm_global = nn.LayerNorm(global_dim)

    def forward(self, x_local, x_global):
        """
        x_local: Tensor of shape (B, L, C_local), where L = H_local * W_local
        x_global: Tensor of shape (B, C_global, H_global, W_global)
        """
        B, L, C_local = x_local.shape
        B_global, C_global, H_global, W_global = x_global.shape

        # Reshape x_global to (B, L_global, C_global)
        x_global = x_global.view(B_global, C_global, -1).transpose(1, 2)  # Shape: (B, L_global, C_global)
        L_global = H_global * W_global

        # Optional: Normalize inputs
        x_local_norm = self.norm_local(x_local)
        x_global_norm = self.norm_global(x_global)

        # Project local features (queries)
        Q = self.query_proj(x_local_norm)  # Shape: (B, L, C_local)

        # Project global features (keys and values)
        K = self.key_proj(x_global_norm)   # Shape: (B, L_global, C_local)
        V = self.value_proj(x_global_norm) # Shape: (B, L_global, C_local)

        # Reshape for multi-head attention
        Q = Q.view(B, L, self.num_heads, self.dim_head_local).transpose(1, 2)  # Shape: (B, num_heads, L, dim_head_local)
        K = K.view(B, L_global, self.num_heads, self.dim_head_local).transpose(1, 2)  # Shape: (B, num_heads, L_global, dim_head_local)
        V = V.view(B, L_global, self.num_heads, self.dim_head_local).transpose(1, 2)  # Shape: (B, num_heads, L_global, dim_head_local)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim_head_local ** 0.5)  # Shape: (B, num_heads, L, L_global)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # Shape: (B, num_heads, L, L_global)

        attn_output = torch.matmul(attn_weights, V)  # Shape: (B, num_heads, L, dim_head_local)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, C_local)  # Shape: (B, L, C_local)

        # Output projection
        attn_output = self.out_proj(attn_output)  # Shape: (B, L, C_local)

        # Residual connection
        x_merged = x_local + attn_output  # Shape: (B, L, C_local)

        return x_merged

class CrossAttentionWithGating(nn.Module):
    def __init__(self, dim, num_heads):
        super(CrossAttentionWithGating, self).__init__()
        self.dim = dim  # Dimensionality of the feature vectors (e.g., 192)
        self.num_heads = num_heads  # Number of attention heads
        self.dim_head = dim // num_heads  # Dimensionality per head

        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(dim, dim)  # Projects local features to queries
        self.k_proj = nn.Linear(dim, dim)  # Projects global features to keys
        self.v_proj = nn.Linear(dim, dim)  # Projects global features to values

        # Linear layer for gating mechanism
        self.gate_proj = nn.Linear(dim * 2, dim)  # Projects concatenated features to gate values

        # Output projection to combine features
        self.out_proj = nn.Linear(dim, dim)  # Projects enhanced features back to original dimension

    def forward(self, local_feat, global_feat):
        """
        local_feat: Tensor of shape (n, num_patches, dim)
        global_feat: Tensor of shape (n, dim, h, w)
        """
        n, num_patches, dim = local_feat.shape
        n, dim, h, w = global_feat.shape

        # Step 1: Flatten the spatial dimensions of global features
        global_feat_flat = global_feat.view(n, dim, h * w).transpose(1, 2)  # Shape: (n, h*w, dim)

        # Step 2: Compute queries from local features
        Q = self.q_proj(local_feat)  # Shape: (n, num_patches, dim)

        # Step 3: Compute keys and values from global features
        K = self.k_proj(global_feat_flat)  # Shape: (n, h*w, dim)
        V = self.v_proj(global_feat_flat)  # Shape: (n, h*w, dim)

        # Step 4: Reshape for multi-head attention
        Q = Q.view(n, num_patches, self.num_heads, self.dim_head).transpose(1, 2)  # (n, num_heads, num_patches, dim_head)
        K = K.view(n, h * w, self.num_heads, self.dim_head).transpose(1, 2)        # (n, num_heads, h*w, dim_head)
        V = V.view(n, h * w, self.num_heads, self.dim_head).transpose(1, 2)        # (n, num_heads, h*w, dim_head)

        # Step 5: Compute scaled dot-product attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim_head)  # (n, num_heads, num_patches, h*w)

        # Step 6: Apply softmax to obtain attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (n, num_heads, num_patches, h*w)

        # Step 7: Compute attention output
        attention_output = torch.matmul(attention_weights, V)  # (n, num_heads, num_patches, dim_head)

        # Step 8: Concatenate attention outputs from all heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(n, num_patches, dim)  # (n, num_patches, dim)

        # Step 9: Compute gating values
        # Concatenate local features and attention output along the feature dimension
        gate_input = torch.cat([local_feat, attention_output], dim=-1)  # Shape: (n, num_patches, dim * 2)
        gate_values = torch.sigmoid(self.gate_proj(gate_input))  # Shape: (n, num_patches, dim)

        # Step 10: Apply gating to modulate the attention output
        gated_attention_output = gate_values * attention_output  # Element-wise multiplication

        # Step 11: Fuse the gated attention output with local features
        enhanced_local_feat = local_feat + gated_attention_output  # Shape: (n, num_patches, dim)

        # Step 12: Apply output projection (optional)
        enhanced_local_feat = self.out_proj(enhanced_local_feat)  # Shape: (n, num_patches, dim)

        return enhanced_local_feat

class CoordinatePositionalEncoding(nn.Module):
    def __init__(self, dim):
        super(CoordinatePositionalEncoding, self).__init__()
        self.dim = dim  # Dimensionality of the positional encodings

    def forward(self, coordinates):
        """
        coordinates: Tensor of shape (n, num_positions, 2), containing normalized (x, y) positions
        Returns positional encodings of shape (n, num_positions, dim)
        """
        n, num_positions, _ = coordinates.shape
        device = coordinates.device
        dtype = coordinates.dtype

        # Ensure positional encoding tensor shape
        pe = torch.zeros(n, num_positions, self.dim, device=device, dtype=dtype)
        
        # Compute division term for sine and cosine functions
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / self.dim)
        )  # Shape: (dim // 2,)
        div_term = div_term.unsqueeze(0).unsqueeze(0)  # Reshape to (1, 1, dim // 2)
        
        # Separate x and y coordinates
        x_coords = coordinates[..., 0].unsqueeze(-1)  # Shape: (n, num_positions, 1)
        y_coords = coordinates[..., 1].unsqueeze(-1)  # Shape: (n, num_positions, 1)

        # Apply sine and cosine to x and y coordinates, broadcasting with div_term
        pe[..., 0::2] = torch.sin(x_coords * div_term)  # Shape: (n, num_positions, dim // 2)
        pe[..., 1::2] = torch.cos(y_coords * div_term)  # Shape: (n, num_positions, dim // 2)

        return pe  # Shape: (n, num_positions, dim)

class CrossAttentionWithPositionalEncoding(nn.Module):
    def __init__(self, dim, num_heads):
        super(CrossAttentionWithPositionalEncoding, self).__init__()
        self.dim = dim  # Dimensionality of the feature vectors
        self.num_heads = num_heads  # Number of attention heads
        self.dim_head = dim // num_heads  # Dimensionality per head

        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

        # Positional encoding module
        self.pos_enc = CoordinatePositionalEncoding(dim)

    def forward(self, local_feat, global_feat):
        """
        local_feat: Tensor of shape (n, N_p, d)
        global_feat: Tensor of shape (n, d, H, W)
        Returns enhanced local features of shape (n, N_p, d)
        """
        n, N_p, d = local_feat.shape
        n, d, H, W = global_feat.shape

        # Step 1: Flatten global features
        global_feat_flat = global_feat.view(n, d, H * W).transpose(1, 2)  # Shape: (n, H*W, d)

        # Step 2: Compute positional coordinates for local and global features
        # Local patch positions
        P_h = P_w = int(math.sqrt(N_p))  # Assuming square grid of patches
        device = local_feat.device

        # Create coordinate grids for local patches
        local_coords = self.get_normalized_coordinates(n, P_h, P_w, device)  # Shape: (n, N_p, 2)

        # Global feature positions
        global_coords = self.get_normalized_coordinates(n, H, W, device)  # Shape: (n, H*W, 2)

        # Step 3: Compute positional encodings
        local_pos_enc = self.pos_enc(local_coords)  # Shape: (n, N_p, d)
        global_pos_enc = self.pos_enc(global_coords)  # Shape: (n, H*W, d)

        # Step 4: Augment local and global features with positional encodings
        local_feat = local_feat + local_pos_enc  # Shape: (n, N_p, d)
        global_feat_flat = global_feat_flat + global_pos_enc  # Shape: (n, H*W, d)

        # Step 5: Compute queries, keys, and values
        Q = self.q_proj(local_feat)  # Shape: (n, N_p, d)
        K = self.k_proj(global_feat_flat)  # Shape: (n, H*W, d)
        V = self.v_proj(global_feat_flat)  # Shape: (n, H*W, d)

        # Step 6: Reshape for multi-head attention
        Q = Q.view(n, N_p, self.num_heads, self.dim_head).transpose(1, 2)  # (n, num_heads, N_p, dim_head)
        K = K.view(n, H * W, self.num_heads, self.dim_head).transpose(1, 2)  # (n, num_heads, H*W, dim_head)
        V = V.view(n, H * W, self.num_heads, self.dim_head).transpose(1, 2)  # (n, num_heads, H*W, dim_head)

        # Step 7: Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim_head)  # (n, num_heads, N_p, H*W)

        # Step 8: Apply softmax to obtain attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (n, num_heads, N_p, H*W)

        # Step 9: Compute attention output
        attention_output = torch.matmul(attention_weights, V)  # (n, num_heads, N_p, dim_head)

        # Step 10: Concatenate attention outputs from all heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(n, N_p, d)  # (n, N_p, d)

        # Step 11: Fuse attention output with local features
        enhanced_local_feat = local_feat + attention_output  # Shape: (n, N_p, d)

        # Step 12: Apply output projection (optional)
        enhanced_local_feat = self.out_proj(enhanced_local_feat)  # Shape: (n, N_p, d)

        return enhanced_local_feat  # Shape: (n, N_p, d)

    def get_normalized_coordinates(self, n, height, width, device):
        """
        Generate normalized coordinates for a grid of size (height, width)
        Returns a tensor of shape (n, height * width, 2) with (x, y) coordinates normalized to [0, 1]
        """
        # Create coordinate grid
        y_coords = torch.linspace(0, 1, steps=height, device=device)
        x_coords = torch.linspace(0, 1, steps=width, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')  # Shape: (height, width)

        # Flatten and stack coordinates
        coords = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)  # Shape: (height*width, 2)
        coords = coords.unsqueeze(0).repeat(n, 1, 1)  # Shape: (n, height*width, 2)

        return coords  # Shape: (n, height*width, 2)
    

class GatedCrossAttentionWithPositionalEncoding(nn.Module):
    def __init__(self, dim, num_heads):
        super(GatedCrossAttentionWithPositionalEncoding, self).__init__()
        self.dim = dim  # Feature dimensionality
        self.num_heads = num_heads  # Number of attention heads
        self.dim_head = dim // num_heads  # Dimensionality per head

        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Linear layer for gating mechanism
        self.gate_proj = nn.Linear(dim * 2, dim)  # Projects concatenated features to gate values

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

        # Positional encoding module
        self.pos_enc = CoordinatePositionalEncoding(dim)

    def forward(self, local_feat, global_feat):
        """
        local_feat: Tensor of shape (n, N_p, d)
        global_feat: Tensor of shape (n, d, H, W)
        Returns enhanced local features of shape (n, N_p, d)
        """
        n, N_p, d = local_feat.shape
        n, d, H, W = global_feat.shape

        # Step 1: Flatten global features
        global_feat_flat = global_feat.view(n, d, H * W).transpose(1, 2)  # Shape: (n, H*W, d)

        # Step 2: Compute positional coordinates
        # Local patch positions
        P_h = P_w = int(math.sqrt(N_p))  # Assuming square grid of patches
        device = local_feat.device

        local_coords = self.get_normalized_coordinates(n, P_h, P_w, device)  # Shape: (n, N_p, 2)
        global_coords = self.get_normalized_coordinates(n, H, W, device)     # Shape: (n, H*W, 2)

        # Step 3: Compute positional encodings
        local_pos_enc = self.pos_enc(local_coords)      # Shape: (n, N_p, d)
        global_pos_enc = self.pos_enc(global_coords)    # Shape: (n, H*W, d)

        # Step 4: Augment features with positional encodings
        local_feat = local_feat + local_pos_enc         # Shape: (n, N_p, d)
        global_feat_flat = global_feat_flat + global_pos_enc  # Shape: (n, H*W, d)

        # Step 5: Compute queries, keys, and values
        Q = self.q_proj(local_feat)                     # Shape: (n, N_p, d)
        K = self.k_proj(global_feat_flat)               # Shape: (n, H*W, d)
        V = self.v_proj(global_feat_flat)               # Shape: (n, H*W, d)

        # Step 6: Reshape for multi-head attention
        Q = Q.view(n, N_p, self.num_heads, self.dim_head).transpose(1, 2)  # (n, num_heads, N_p, dim_head)
        K = K.view(n, H * W, self.num_heads, self.dim_head).transpose(1, 2)  # (n, num_heads, H*W, dim_head)
        V = V.view(n, H * W, self.num_heads, self.dim_head).transpose(1, 2)  # (n, num_heads, H*W, dim_head)

        # Step 7: Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim_head)  # (n, num_heads, N_p, H*W)

        # Step 8: Apply softmax to obtain attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (n, num_heads, N_p, H*W)

        # Step 9: Compute attention output
        attention_output = torch.matmul(attention_weights, V)  # (n, num_heads, N_p, dim_head)

        # Step 10: Concatenate attention outputs from all heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(n, N_p, d)  # (n, N_p, d)

        # Step 11: Compute gating values
        gate_input = torch.cat([local_feat, attention_output], dim=-1)  # Shape: (n, N_p, 2d)
        gate_values = torch.sigmoid(self.gate_proj(gate_input))         # Shape: (n, N_p, d)

        # Step 12: Apply gating to modulate the attention output
        gated_attention_output = gate_values * attention_output         # Shape: (n, N_p, d)

        # Step 13: Fuse attention output with local features
        enhanced_local_feat = local_feat + gated_attention_output       # Shape: (n, N_p, d)

        # Step 14: Apply output projection
        enhanced_local_feat = self.out_proj(enhanced_local_feat)        # Shape: (n, N_p, d)

        return enhanced_local_feat

    def get_normalized_coordinates(self, n, height, width, device):
        """
        Generate normalized coordinates for a grid of size (height, width)
        Returns a tensor of shape (n, height * width, 2) with (x, y) coordinates normalized to [0, 1]
        """
        # Create coordinate grid
        y_coords = torch.linspace(0, 1, steps=height, device=device)
        x_coords = torch.linspace(0, 1, steps=width, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')  # Shape: (height, width)

        # Flatten and stack coordinates
        coords = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)  # Shape: (height*width, 2)
        coords = coords.unsqueeze(0).repeat(n, 1, 1)  # Shape: (n, height*width, 2)

        return coords  # Shape: (n, height*width, 2)
    
class RoPEMultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(RoPEMultiheadAttention, self).__init__()
        self.dim = dim  # Feature dimensionality
        self.num_heads = num_heads  # Number of attention heads
        self.dim_head = dim // num_heads  # Dimensionality per head

        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Linear layer for gating mechanism
        self.gate_proj = nn.Linear(dim * 2, dim)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, local_feat, global_feat):
        """
        local_feat: Tensor of shape (n, N_p, d)
        global_feat: Tensor of shape (n, d, H, W)
        Returns enhanced local features of shape (n, N_p, d)
        """
        n, N_p, d = local_feat.shape
        n, d, H, W = global_feat.shape

        # Step 1: Flatten global features
        global_feat_flat = global_feat.view(n, d, H * W).transpose(1, 2)  # Shape: (n, H*W, d)

        # Step 2: Compute positional coordinates
        # Local patch positions
        P_h = P_w = int(math.sqrt(N_p))  # Assuming square grid of patches
        device = local_feat.device

        local_coords = self.get_normalized_coordinates(n, P_h, P_w, device)  # Shape: (n, N_p, 2)
        global_coords = self.get_normalized_coordinates(n, H, W, device)     # Shape: (n, H*W, 2)

        # Step 3: Compute queries, keys, and values
        Q = self.q_proj(local_feat)                     # Shape: (n, N_p, d)
        K = self.k_proj(global_feat_flat)               # Shape: (n, H*W, d)
        V = self.v_proj(global_feat_flat)               # Shape: (n, H*W, d)

        # Step 4: Reshape for multi-head attention
        Q = Q.view(n, N_p, self.num_heads, self.dim_head).transpose(1, 2)  # (n, num_heads, N_p, dim_head)
        K = K.view(n, H * W, self.num_heads, self.dim_head).transpose(1, 2)  # (n, num_heads, H*W, dim_head)
        V = V.view(n, H * W, self.num_heads, self.dim_head).transpose(1, 2)  # (n, num_heads, H*W, dim_head)

        # Step 5: Apply RoPE to queries and keys
        Q = self.apply_rope(Q, local_coords)            # (n, num_heads, N_p, dim_head)
        K = self.apply_rope(K, global_coords)           # (n, num_heads, H*W, dim_head)

        # Step 6: Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim_head)  # (n, num_heads, N_p, H*W)

        # Step 7: Apply softmax to obtain attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (n, num_heads, N_p, H*W)

        # Step 8: Compute attention output
        attention_output = torch.matmul(attention_weights, V)    # (n, num_heads, N_p, dim_head)

        # Step 9: Concatenate attention outputs from all heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(n, N_p, d)  # (n, N_p, d)

        # Step 10: Compute gating values
        gate_input = torch.cat([local_feat, attention_output], dim=-1)  # Shape: (n, N_p, 2d)
        gate_values = torch.sigmoid(self.gate_proj(gate_input))         # Shape: (n, N_p, d)

        # Step 11: Apply gating to modulate the attention output
        gated_attention_output = gate_values * attention_output         # Shape: (n, N_p, d)

        # Step 12: Fuse attention output with local features
        enhanced_local_feat = local_feat + gated_attention_output       # Shape: (n, N_p, d)

        # Step 13: Apply output projection
        enhanced_local_feat = self.out_proj(enhanced_local_feat)        # Shape: (n, N_p, d)

        return enhanced_local_feat

    def get_normalized_coordinates(self, n, height, width, device):
        """
        Generate normalized coordinates for a grid of size (height, width)
        Returns a tensor of shape (n, height * width, 2) with (x, y) coordinates normalized to [0, 1]
        """
        # Create coordinate grid
        y_coords = torch.linspace(0, 1, steps=height, device=device)
        x_coords = torch.linspace(0, 1, steps=width, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')  # Shape: (height, width)

        # Flatten and stack coordinates
        coords = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)  # Shape: (height*width, 2)
        coords = coords.unsqueeze(0).repeat(n, 1, 1)  # Shape: (n, height*width, 2)

        return coords  # Shape: (n, height*width, 2)

    def apply_rope(self, tensor, coords):
        """
        Apply Rotary Positional Embeddings (RoPE) to the tensor based on the positions.
        tensor: Shape (n, num_heads, seq_len, dim_head)
        coords: Shape (n, seq_len, 2)
        Returns tensor of the same shape with RoPE applied.
        """
        n, num_heads, seq_len, dim_head = tensor.shape
        device = tensor.device

        # Ensure dim_head is even
        assert dim_head % 2 == 0, "dim_head must be even for RoPE"

        # Compute position frequencies
        dim_half = dim_head // 2
        freq_seq = torch.arange(0, dim_half, device=device, dtype=torch.float32)
        freq_seq = 1.0 / (10000 ** (freq_seq / dim_half))  # Shape: (dim_half,)

        # Compute angles based on coordinates
        coords = coords.unsqueeze(1)  # Shape: (n, 1, seq_len, 2)
        angles_x = coords[..., 0:1] * freq_seq  # Shape: (n, 1, seq_len, dim_half)
        angles_y = coords[..., 1:2] * freq_seq  # Shape: (n, 1, seq_len, dim_half)

        # Concatenate angles along the last dimension
        angles = torch.cat([angles_x, angles_y], dim=-1)  # Shape: (n, 1, seq_len, dim_head)

        # Compute rotation matrices (cosine and sine components)
        cos_angles = torch.cos(angles)  # Shape: (n, 1, seq_len, dim_head)
        sin_angles = torch.sin(angles)  # Shape: (n, 1, seq_len, dim_head)

        # Apply rotations to the tensor
        tensor_rotated = (tensor * cos_angles) + (self.rotate_half(tensor) * sin_angles)

        return tensor_rotated  # Shape: (n, num_heads, seq_len, dim_head)

    def rotate_half(self, x):
        """
        Rotate the tensor by swapping and negating halves.
        x: Tensor of shape (..., dim_head)
        Returns rotated tensor of the same shape.
        """
        dim_head = x.shape[-1]
        dim_half = dim_head // 2
        x1 = x[..., :dim_half]  # First half
        x2 = x[..., dim_half:]  # Second half
        return torch.cat([-x2, x1], dim=-1)  # Rotate by 90 degrees in the plane


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SUNet(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3

        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, out_chans=3,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="Dual up-sample", **kwargs):
        super(SUNet, self).__init__()

        self.out_chans = out_chans
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.prelu = nn.PReLU()
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        self.gc_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)


        '''
        self.bottleneck = Bottleneck(
            channels=int(embed_dim * 2 ** (i_layer)), 
            block=BasicLayer(dim=int(embed_dim * 2 ** (i_layer)),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample= None,
                               use_checkpoint=use_checkpoint)
        )
        '''

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                              self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = UpSample(input_resolution=patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                    in_channels=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), scale_factor=2)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                             patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                             patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=UpSample if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "Dual up-sample":
            self.up = UpSample(input_resolution=(img_size // patch_size, img_size // patch_size),
                               in_channels=embed_dim, scale_factor=4)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.out_chans, kernel_size=3, stride=1,
                                    padding=1, bias=False)  # kernel = 1

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        residual = x
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C

        return x, residual, x_downsample

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)  # concat last dimension
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C

        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "Dual up-sample":
            x = self.up(x)
            # x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W

        return x

    def forward(self, x):
        #print(f'initial {x.shape}')

        x = self.conv_first(x) # obtain first feature map

        #print(f'after frist conv {x.shape}')
        x, residual, x_downsample = self.forward_features(x)
        x = self.bottleneck(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)
        out = self.output(x)
        # x = x + residual
        return out

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.out_chans
        return flops
