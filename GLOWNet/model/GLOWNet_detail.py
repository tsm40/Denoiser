import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import to_2tuple, trunc_normal_
from .bottleneck import Bottleneck
from .cross_attn import *
from .swin_basiclayer import *
from .gc_basiclayer import *

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

class BasicLayerWithContext(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, gc_downsample=None, use_checkpoint=False,
                 context_ratio=1./16, context_pooling_type='att', context_fusion_types=('channel_add', ),
                 cross_attn_type='CrossAttentionLayer', cross_attn_args=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Calculate context block depth (1/3 of Swin blocks' depth)
        self.context_depth = max(1, depth // 3)

        # Build Swin Transformer blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            ) for i in range(depth)
        ])

        # Swin blocks' downsampling layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        # Build ContextBlocks
        self.context_blocks = nn.ModuleList([
            ContextBlock(
                inplanes=dim,
                ratio=context_ratio,
                pooling_type=context_pooling_type,
                fusion_types=context_fusion_types
            ) for _ in range(self.context_depth)
        ])

        if gc_downsample is not None:
            self.gc_downsample = gc_downsample(dim, dim * 2)
        else:
            self.gc_downsample = None

        # Cross-Attention Layer Selection
        cross_attn_classes = {
            'CrossAttentionLayer': CrossAttentionLayer,
            'CrossAttentionWithGating': CrossAttentionWithGating,
            'CrossAttentionWithPositionalEncoding': CrossAttentionWithPositionalEncoding,
            'GatedCrossAttentionWithPositionalEncoding': GatedCrossAttentionWithPositionalEncoding,
            'RoPEMultiheadAttention': RoPEMultiheadAttention,
        }

        assert cross_attn_type in cross_attn_classes, f"Invalid cross attention type: {cross_attn_type}"
        cross_attn_class = cross_attn_classes[cross_attn_type]

        if cross_attn_args is None:
            cross_attn_args = {}
        # Ensure necessary arguments are included
        cross_attn_dim = dim 
        if downsample:
            cross_attn_dim = cross_attn_dim * 2
        cross_attn_args.setdefault('dim', cross_attn_dim) # downsampled
        cross_attn_args.setdefault('num_heads', num_heads)

        # Initialize cross-attention layer
        self.cross_attention = cross_attn_class(**cross_attn_args)

    def forward(self, x, gc_x):
        # Process input through Swin Transformer blocks
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)

        # Process through ContextBlocks
        for cb in self.context_blocks:
            if self.use_checkpoint:
                gc_x = checkpoint.checkpoint(cb, gc_x)
            else:
                gc_x = cb(gc_x)
        if self.gc_downsample is not None:
            gc_x = self.gc_downsample(gc_x)
        # print(f"Context Layer Down gc_x {gc_x.shape}")
        # Apply Cross-Attention Layer
        x = self.cross_attention(x, gc_x)

        return x, gc_x

class BasicLayerUpWithContext(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, gc_upsample = None, use_checkpoint=False,
                 context_ratio=1./16, context_pooling_type='att', context_fusion_types=('channel_add', ),
                 cross_attn_type='CrossAttentionLayer', cross_attn_args=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Calculate context block depth (1/3 of Swin blocks' depth)
        self.context_depth = max(1, depth // 3)

        # Build Swin Transformer blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            ) for i in range(depth)
        ])

        # Swin blocks' upsampling layer
        if upsample is not None:
            self.upsample = UpSample(input_resolution, in_channels=dim, scale_factor=2)
        else:
            self.upsample = None

        # Build ContextBlocks
        self.context_blocks = nn.ModuleList([
            ContextBlock(
                inplanes=dim,
                ratio=context_ratio,
                pooling_type=context_pooling_type,
                fusion_types=context_fusion_types
            ) for _ in range(self.context_depth)
        ])

        # ContextBlocks' upsampling layer
        if gc_upsample is not None:
            self.context_upsample = gc_upsample(in_channels=dim, scale_factor=2)
        else:
            self.context_upsample = None

        # Cross-Attention Layer Selection
        cross_attn_classes = {
            'CrossAttentionLayer': CrossAttentionLayer,
            'CrossAttentionWithGating': CrossAttentionWithGating,
            'CrossAttentionWithPositionalEncoding': CrossAttentionWithPositionalEncoding,
            'GatedCrossAttentionWithPositionalEncoding': GatedCrossAttentionWithPositionalEncoding,
            'RoPEMultiheadAttention': RoPEMultiheadAttention
        }

        assert cross_attn_type in cross_attn_classes, f"Invalid cross attention type: {cross_attn_type}"
        cross_attn_class = cross_attn_classes[cross_attn_type]

        if cross_attn_args is None:
            cross_attn_args = {}

        cross_attn_dim = dim 
        if upsample:
            cross_attn_dim = cross_attn_dim // 2
        cross_attn_args.setdefault('dim', cross_attn_dim) # downsampled
        cross_attn_args.setdefault('num_heads', num_heads)

        # Initialize cross-attention layer
        self.cross_attention = cross_attn_class(**cross_attn_args)

    def forward(self, x, gc_x):
        # Process input through Swin Transformer blocks
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)

        # Process through ContextBlocks
        for cb in self.context_blocks:
            if self.use_checkpoint:
                gc_x = checkpoint.checkpoint(cb, gc_x)
            else:
                gc_x = cb(gc_x)
        if self.context_upsample is not None:
            gc_x = self.context_upsample(gc_x)
        
        # print(f"Context Layer Up gc_x {gc_x.shape}")
        # Apply Cross-Attention Layer
        x = self.cross_attention(x, gc_x)

        return x, gc_x
    
class GLOWNet(nn.Module):
    r""" Swin Transformer UNet (SUNet)
        A PyTorch implementation that integrates Swin Transformer blocks with context blocks and cross-attention layers.

    Args:
        img_size (int | tuple(int)): Input image size. Default: 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        out_chans (int): Number of output image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use gradient checkpointing to save memory. Default: False
        final_upsample (str): Final upsampling method. Default: "Dual up-sample"
        context_ratio (float): Ratio parameter for ContextBlock. Default: 1./16
        context_pooling_type (str): Pooling type for ContextBlock ('att' or 'avg'). Default: 'att'
        context_fusion_types (tuple(str)): Fusion types for ContextBlock. Default: ('channel_add', )
        cross_attn_type (str): Type of cross-attention layer to use. Default: 'CrossAttentionLayer'
        cross_attn_args (dict): Additional arguments for the cross-attention layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, out_chans=3,
                 embed_dim=96, depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="Dual up-sample",
                 context_ratio=1./16, context_pooling_type='att', context_fusion_types=('channel_add', ),
                 cross_attn_type='CrossAttentionLayer', cross_attn_args=None, **kwargs):
        super(GLOWNet, self).__init__()

        self.out_chans = out_chans
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.gc_conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

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
            layer = BasicLayerWithContext(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer)
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                gc_downsample=DownsamplingBlock if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                # New arguments for context blocks
                context_ratio=context_ratio,
                context_pooling_type=context_pooling_type,
                context_fusion_types=context_fusion_types,
                # New arguments for cross-attention layers
                cross_attn_type=cross_attn_type,
                cross_attn_args=cross_attn_args
            )
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
        self.gc_up = GCUpSample(
            in_channels=int(embed_dim * 2 ** (self.num_layers - 1)),
            scale_factor=2)
        self.concat_back_dim = nn.ModuleList()
        self.gc_concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(
                2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))
            ) if i_layer > 0 else nn.Identity()
            
            gc_concat = nn.Conv2d(
                2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), 3, 1, 1
            ) if i_layer > 0 else nn.Identity()

            if i_layer == 0:
                # For the first layer, use UpSample directly
                layer_up = UpSample(
                    input_resolution=patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                    in_channels=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    scale_factor=2
                )
            else:
                # For subsequent layers, use BasicLayerUpWithContext
                layer_up = BasicLayerUpWithContext(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))
                    ),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[:(self.num_layers - 1 - i_layer)]):
                        sum(depths[:(self.num_layers - 1 - i_layer) + 1])
                    ],
                    norm_layer=norm_layer,
                    upsample=UpSample if (i_layer < self.num_layers - 1) else None,
                    gc_upsample=GCUpSample if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                    # New arguments for context blocks
                    context_ratio=context_ratio,
                    context_pooling_type=context_pooling_type,
                    context_fusion_types=context_fusion_types,
                    # New arguments for cross-attention layers
                    cross_attn_type=cross_attn_type,
                    cross_attn_args=cross_attn_args
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)
            self.gc_concat_back_dim.append(gc_concat)

        self.norm = norm_layer(self.num_features)
        self.gc_norm = nn.BatchNorm2d(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)
        self.gc_norm_up = nn.BatchNorm2d(self.embed_dim)

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
    def forward_features(self, x, gc_x):
        residual = x
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        gc_x = self.pos_drop(gc_x)
        x_downsample = []
        gc_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            gc_downsample.append(gc_x)
            x, gc_x = layer(x, gc_x)

        x = self.norm(x)  # B L C
        gc_x = self.gc_norm(gc_x)
        return x, x_downsample, gc_x, gc_downsample

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample, gc_x, gc_downsample):
        #print(f'x {x.shape} gc {gc_x.shape}')
        
        #for d, gcd in zip(x_downsample, gc_downsample):
        #    print(f'd {d.shape} gc_d {gcd.shape}')
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
                gc_x = self.gc_up(gc_x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)  # concat last dimension
                gc_x = torch.cat([gc_x, gc_downsample[3 - inx]], 1)
                x = self.concat_back_dim[inx](x)
                gc_x = self.gc_concat_back_dim[inx](gc_x)
                x, gc_x = layer_up(x, gc_x)

        x = self.norm_up(x)  # B L C
        gc_x = self.gc_norm_up(gc_x)
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
        gc = x 
        x= self.conv_first(x)
        gc_x = self.gc_conv_first(gc) # obtain first feature map

        #print(f'after frist conv {x.shape}')
        x, x_downsample, gc_x, gc_downsample = self.forward_features(x, gc_x)
        #x = self.bottleneck(x)
        x = self.forward_up_features(x, x_downsample, gc_x, gc_downsample)
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
