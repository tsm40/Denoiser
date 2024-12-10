import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from thop import profile
from mmcv.cnn import constant_init, kaiming_init
import torch.nn.functional as F


# global context network
def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)

class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out

class ConvDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv_down = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.norm = norm_layer(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_down(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class AttentionDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(AttentionDownsample, self).__init__()
        # Channel attention mechanism
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )

        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride
            ),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        attention_map = self.channel_attention(x)  # (B, 1, H, W)
        x_attended = x * attention_map
        x_downsampled = self.conv(x_attended)
        res = self.residual(x)
        return F.relu(x_downsampled + res)

class ShapeTransform:
    def __init__(self, input_resolution):
        self.input_resolution = input_resolution
    
    def to_4d(self, x):
        """Convert (B, L, C) to (B, C, H, W)"""
        B, L, C = x.shape
        H, W = self.input_resolution
        return x.view(B, H, W, C).permute(0, 3, 1, 2)
    
    def to_3d(self, x):
        """Convert (B, C, H, W) to (B, L, C)"""
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).view(B, H*W, C)

# Dual up-sample
class UpSample(nn.Module):
    def __init__(self, input_resolution, in_channels, scale_factor):
        super(UpSample, self).__init__()
        self.input_resolution = input_resolution
        self.factor = scale_factor


        if self.factor == 2:
            self.conv = nn.Conv2d(in_channels, in_channels//2, 1, 1, 0, bias=False)
            self.up_p = nn.Sequential(nn.Conv2d(in_channels, 2*in_channels, 1, 1, 0, bias=False),
                                      nn.PReLU(),
                                      nn.PixelShuffle(scale_factor),
                                      nn.Conv2d(in_channels//2, in_channels//2, 1, stride=1, padding=0, bias=False))

            self.up_b = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                                      nn.PReLU(),
                                      nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                                      nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=False))
        elif self.factor == 4:
            self.conv = nn.Conv2d(2*in_channels, in_channels, 1, 1, 0, bias=False)
            self.up_p = nn.Sequential(nn.Conv2d(in_channels, 16 * in_channels, 1, 1, 0, bias=False),
                                      nn.PReLU(),
                                      nn.PixelShuffle(scale_factor),
                                      nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

            self.up_b = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                                      nn.PReLU(),
                                      nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                                      nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        """
        x: B, L = H*W, C
        """
        if type(self.input_resolution) == int:
            H = self.input_resolution
            W = self.input_resolution

        elif type(self.input_resolution) == tuple:
            H, W = self.input_resolution

        B, L, C = x.shape
        x = x.view(B, H, W, C)  # B, H, W, C
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        x_p = self.up_p(x)  # pixel shuffle
        x_b = self.up_b(x)  # bilinear
        out = self.conv(torch.cat([x_p, x_b], dim=1))
        out = out.permute(0, 2, 3, 1)  # B, H, W, C
        if self.factor == 2:
            out = out.view(B, -1, C // 2)

        return out

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, context_ratio, pooling_type, fusion_types,
                 depth, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # GC parameters
        self.context_ratio = context_ratio
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        # Create shape transform helper
        self.shape_transform = ShapeTransform(input_resolution)

        # create a series of ContextBlocks
        self.blocks = nn.ModuleList([
            ContextBlock(inplanes=dim, ratio=context_ratio, pooling_type=pooling_type,
                         fusion_types=fusion_types) for _ in range(depth)
        ])

        # downsample layer
        if downsample is not None:
            self.downsample = downsample(in_channels=dim, out_channels=2*dim)
        else:
            self.downsample = None

    def forward(self, x):       
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = self.shape_transform.to_3d(
                    blk(self.shape_transform.to_4d(x))
                )

        # Apply downsample
        if self.downsample is not None:
            x = self.shape_transform.to_3d(
                self.downsample(self.shape_transform.to_4d(x))
            )
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class BasicLayer_up(nn.Module):
    def __init__(self, dim, input_resolution, context_ratio, pooling_type, fusion_types,
                 depth, norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # GC parameters
        self.context_ratio = context_ratio
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        # Create shape transform helper
        self.shape_transform = ShapeTransform(input_resolution)

        # create a series of ContextBlocks
        self.blocks = nn.ModuleList([
            ContextBlock(inplanes=dim, ratio=context_ratio, pooling_type=pooling_type,
                         fusion_types=fusion_types) for _ in range(depth)
        ])

        # upsample
        if upsample is not None:
            self.upsample = UpSample(input_resolution, in_channels=dim, scale_factor=2)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = self.shape_transform.to_3d(
                    blk(self.shape_transform.to_4d(x))
                )
                
        if self.upsample is not None:
            x = self.upsample(x)
        return x

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

class GCNet(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, out_chans=3,
                 embed_dim=96, depths=[2, 2, 2, 2],
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, final_upsample="Dual up-sample",
                 context_ratio=1./16, pooling_type='att', fusion_types=('channel_add', ), **kwargs):
        super(GCNet, self).__init__()

        self.out_chans = out_chans
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.final_upsample = final_upsample
        self.prelu = nn.PReLU()
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.context_ratio = context_ratio
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               norm_layer=norm_layer,
                               downsample=AttentionDownsample if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               # gc parameters
                               context_ratio=context_ratio, 
                               pooling_type=pooling_type, 
                               fusion_types=fusion_types)
            self.layers.append(layer)

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
                                         norm_layer=norm_layer,
                                         upsample=UpSample if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint,
                                         # gc parameters
                                         context_ratio=context_ratio, 
                                         pooling_type=pooling_type, 
                                         fusion_types=fusion_types)
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


    # Encoder and Bottleneck
    def forward_features(self, x):
        residual = x
        x = self.patch_embed(x)
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
        x = self.conv_first(x)
        x, residual, x_downsample = self.forward_features(x)
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


if __name__ == '__main__':
    from utils.model_utils import network_parameters

    height = 64
    width = 64
    x = torch.randn((1, 3, height, width))  # .cuda()
    model = GCNet(img_size=256, patch_size=4, in_chans=3, out_chans=3,
                  embed_dim=96, depths=[8, 8, 8, 8],
                  norm_layer=nn.LayerNorm, patch_norm=True,
                  use_checkpoint=False, final_upsample="Dual up-sample",
                  context_ratio=1./16, pooling_type='att', fusion_types=('channel_add', ))  # .cuda()
    # print(model)
    print('input image size: (%d, %d)' % (height, width))
    print('FLOPs: %.4f G' % (model.flops() / 1e9))
    print('model parameters: ', network_parameters(model))
    # x = model(x)
    print('output image size: ', x.shape)
    flops, params = profile(model, (x,))
    print(flops)
    print(params)