import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(CrossAttentionLayer, self).__init__()

        local_dim, global_dim = dim, dim 
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
        pool = torch.nn.AdaptiveAvgPool2d((H_global//4, W_global//4))
        x_global = pool(x_global)
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
        B_global, C_global, H_global, W_global = global_feat.shape
        pool = torch.nn.AdaptiveAvgPool2d((H_global//4, W_global//4))
        global_feat = pool(global_feat)
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

        #print(f'Q {Q.shape} K {K.shape} V {V.shape}')
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
        B_global, C_global, H_global, W_global = global_feat.shape
        pool = torch.nn.AdaptiveAvgPool2d((H_global//4, W_global//4))
        global_feat = pool(global_feat)
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
        B_global, C_global, H_global, W_global = global_feat.shape
        pool = torch.nn.AdaptiveAvgPool2d((H_global//4, W_global//4))
        global_feat = pool(global_feat)
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

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, local_feat, global_feat):
        """
        local_feat: Tensor of shape (n, N_p, d)
        global_feat: Tensor of shape (n, d, H, W)
        Returns enhanced local features of shape (n, N_p, d)
        """
        n, N_p, d = local_feat.shape
        B_global, C_global, H_global, W_global = global_feat.shape
        pool = torch.nn.AdaptiveAvgPool2d((H_global//4, W_global//4))
        global_feat = pool(global_feat)
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

        enhanced_local_feat = local_feat + attention_output       # Shape: (n, N_p, d)

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

