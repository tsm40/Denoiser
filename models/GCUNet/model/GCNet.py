import torch.nn as nn
from model.GCNet_detail_no_pe import GCNet


class GCNet_model(nn.Module):
    def __init__(self, config):
        super(GCNet_model, self).__init__()
        self.config = config
        self.gc_net = GCNet(img_size=config['GCNET']['IMG_SIZE'],
                               patch_size=config['GCNET']['PATCH_SIZE'],
                               in_chans=3,
                               out_chans=3,
                               embed_dim=config['GCNET']['EMB_DIM'],
                               depths=config['GCNET']['DEPTH_EN'],
                               patch_norm=config['GCNET']['PATCH_NORM'],
                               use_checkpoint=config['GCNET']['USE_CHECKPOINTS'],
                               context_ratio=config['GCNET']['CONTEXT_RATIO'],
                               pooling_type=config['GCNET']['POOLING_TYPE'],
                               fusion_types=config['GCNET']['FUSION_TYPES'],
                               )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.gc_net(x)
        return logits
    
if __name__ == '__main__':
    from utils.model_utils import network_parameters
    import torch
    import yaml
    from thop import profile
    from utils.model_utils import network_parameters

    ## Load yaml configuration file
    with open('../training.yaml', 'r') as config:
        opt = yaml.safe_load(config)
    Train = opt['TRAINING']
    OPT = opt['OPTIM']

    height = 256
    width = 256
    x = torch.randn((1, 156, height, width))  # .cuda()
    model = GCNet_model(opt)  # .cuda()
    out = model(x)
    flops, params = profile(model, (x,))
    print(out.size())
    print(flops)
    print(params)
