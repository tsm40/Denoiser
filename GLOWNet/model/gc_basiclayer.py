import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.checkpoint as checkpoint

def kaiming_init(module, mode='fan_in', nonlinearity='relu'):
    if isinstance(module, nn.Conv2d):
        init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=nonlinearity)
        if module.bias is not None:
            init.zeros_(module.bias)

def constant_init(module, val=0):
    if hasattr(module, 'weight') and module.weight is not None:
        init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, val)

def last_zero_init(module):
    if isinstance(module, nn.Sequential) and len(module) > 0:
        constant_init(module[-1], val=0)
    elif module is not None:
        constant_init(module, val=0)

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
        inplanes = 96                 # TEMP HARDCODING INPLANES
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            #print('INPLANES:', inplanes)
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
        #print('gc spatial pooling. xsize:', x.size())
        #x = x.permute(2,0,1)
        batch, channel, height, width = x.size()
        L = height * width
        #batch, channel, L = x.size()
        if self.pooling_type == 'att':
            #print('self.pooling_type==att. inside if statement')
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, L)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, L)
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
        #print('gc forward.')
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

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(DownsamplingBlock, self).__init__()
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
    
class GCUpSample(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super(GCUpSample, self).__init__()
        self.scale_factor = scale_factor

        if self.scale_factor == 2:
            # Output channels reduced by half
            out_channels = in_channels // 2

            # Pixel Shuffle upsampling path
            self.up_p = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * (scale_factor), kernel_size=1, stride=1, padding=0, bias=False),
                nn.PReLU(),
                nn.PixelShuffle(self.scale_factor),  # Upsamples H and W by scale_factor
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )

            # Bilinear upsampling path
            self.up_b = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.PReLU(),
                nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )

            # Final convolution to merge the paths
            self.conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        elif self.scale_factor == 4:
            # Output channels reduced by a factor of 4
            out_channels = in_channels // 4

            # Pixel Shuffle upsampling path
            self.up_p = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * (scale_factor), kernel_size=1, stride=1, padding=0, bias=False),
                nn.PReLU(),
                nn.PixelShuffle(self.scale_factor),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )

            # Bilinear upsampling path
            self.up_b = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.PReLU(),
                nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )

            # Final convolution to merge the paths
            self.conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        else:
            raise ValueError("Unsupported scale factor. Only 2 and 4 are supported.")

    def forward(self, x):
        """
        x: Tensor of shape (B, C_in, H_in, W_in)
        """
        # Apply Pixel Shuffle upsampling
        x_p = self.up_p(x)
        # Apply Bilinear upsampling
        x_b = self.up_b(x)
        # Concatenate along the channel dimension
        out = torch.cat([x_p, x_b], dim=1)
        # Merge the features
        out = self.conv(out)
        return out  # Output shape: (B, C_out, H_out, W_out)
    
class GlobalContextBasicLayer(nn.Module):
    def __init__(self, dim, depth, downsample=None, norm_layer=nn.BatchNorm2d, use_checkpoint=False):
        #print('we making a gc basic layer')
        super(GlobalContextBasicLayer, self).__init__()
        self.dim = dim
        depth = 2      # TEMP HARDCODING DEPTH
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        #self.norm = nn.BatchNorm2d(96)  # inplanes

        # Create a series of ContextBlocks
        self.blocks = nn.ModuleList([
            ContextBlock(inplanes=dim, ratio=1.0) for _ in range(depth)
        ])
        
        if downsample is not None:
            self.downsample = downsample(dim, dim*2, norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        #print('we going forward in a gc basic layer')
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x

class GlobalContextBasicLayer_up(nn.Module):
    def __init__(self, dim, input_resolution, depth, use_checkpoint=False, upsample=None):
        super(GlobalContextBasicLayer_up, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Context blocks for upsampling layers
        self.blocks = nn.ModuleList([
            ContextBlock(inplanes=dim, ratio=1.0) for _ in range(depth)
        ])

        if upsample is not None:
            self.upsample = GCUpSample(in_channels=dim, scale_factor=2)
        else:
            self.upsample = None

    def forward(self, x):
        #print('we are upsampling gc basic layer')
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x