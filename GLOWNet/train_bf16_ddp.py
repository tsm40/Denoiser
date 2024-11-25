# OS and Argument Parsing
import os

# PyTorch modules for distributed training and parallel processing
import torch
import yaml
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import time
import utils
from utils import network_parameters
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random, logging
from data_RGB import get_training_data, get_validation_data

from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from model.SUNet import SUNet_model, GLOWNet_model

# Import autocast and GradScaler for mixed precision
from torch.cuda.amp import autocast, GradScaler

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def main(opt, Train, OPT):
    
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    
    seed = 327 * dist.get_world_size() + rank 
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    mode = opt['MODEL']['MODE']
    log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')
    utils.mkdir(log_dir)
    logger = create_logger(log_dir)
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    model_restored = GLOWNet_model(opt).to(device)
    model_restored = DDP(model_restored, device_ids=[rank], find_unused_parameters=True)

    cnt = 0
    for name, param in model_restored.named_parameters():
        if param.grad is None:
            if cnt >= 20:
                break
            logger.info(f"Parameter {name} was not used in the forward pass.")
            cnt += 1

    p_number = network_parameters(model_restored)
    logger.info(f"Total {p_number}, {cnt} not used in forward")
    ## Training model path direction
    mode = opt['MODEL']['MODE']

    model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
    utils.mkdir(model_dir)
    train_dir = Train['TRAIN_DIR']
    train_dataset = get_training_data(train_dir, {'patch_size': Train['TRAIN_PS']})
    val_dir = Train['VAL_DIR']
    val_dataset = get_validation_data(val_dir, {'patch_size': Train['VAL_PS']})


    world_size=2
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=int(OPT['BATCH'] // world_size),
        shuffle=False,
        num_workers=2,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=False)

    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=2,
        sampler=val_sampler,
        pin_memory=True,
        drop_last=False)
    ## Optimizer
    start_epoch = 1
    new_lr = float(OPT['LR_INITIAL'])
    optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

    ## Scheduler (Strategy)
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                            eta_min=float(OPT['LR_MIN']))
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    #scheduler.step()
    L1_loss = nn.L1Loss()

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    best_psnr = 0
    best_ssim = 0
    best_epoch_psnr = 0
    best_epoch_ssim = 0
    total_start_time = time.time()

    log_steps = 0
    train_steps = 0
    running_loss = 0 

        # Show the training configuration
    logger.info(f'''==> Training details:
    ------------------------------------------------------------------
        Restoration mode:   {mode}
        Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
        Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
        Model parameters:   {p_number}
        Attn Method:        {(opt['SWINUNET']['CROSS_ATTN_TYPE'])}
        Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
        Batch sizes:        {OPT['BATCH']}
        Learning rate:      {OPT['LR_INITIAL']}''')
    logger.info('------------------------------------------------------------------')

    for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        model_restored.train()
        for i, data in enumerate(tqdm(train_loader), 0):
            # Clear gradients
            target = data[0].to(device)
            input_ = data[1].to(device)
            with autocast(dtype=torch.bfloat16):
                restored = model_restored(input_)
                # Compute loss
                loss = L1_loss(restored, target)
            # Backward pass and optimization
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

        ## Evaluation (Validation)
        if epoch % Train['VAL_AFTER_EVERY'] == 0:
            model_restored.eval()
            psnr_val_rgb = []
            ssim_val_rgb = []
            for ii, data_val in enumerate(val_loader, 0):
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()
                with torch.no_grad():
                    with autocast(dtype=torch.bfloat16):
                        restored = model_restored(input_)

                restored = restored.to(target.dtype)
                for res, tar in zip(restored, target):
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))
                    ssim_val_rgb.append(utils.torchSSIM(restored, target))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
            ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()

            # Save the best PSNR model of validation
            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch_psnr = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restored.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(model_dir, "model_bestPSNR.pth"))
            logger.info("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
                epoch, psnr_val_rgb, best_epoch_psnr, best_psnr))

            # Save the best SSIM model of validation
            if ssim_val_rgb > best_ssim:
                best_ssim = ssim_val_rgb
                best_epoch_ssim = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restored.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(model_dir, "model_bestSSIM.pth"))
            
            logger.info("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
                epoch, ssim_val_rgb, best_epoch_ssim, best_ssim))
            dist.barrier()

        scheduler.step()
        torch.cuda.synchronize()
        # Reduce loss history over all processes:
        avg_loss = torch.tensor(running_loss / log_steps, device=device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss.item() / dist.get_world_size()
        logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}")

        # Reset monitoring variables:
        running_loss = 0
        log_steps = 0
        logger.info("------------------------------------------------------------------")
        logger.info("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                                epoch_loss, scheduler.get_lr()[0]))
        logger.info("------------------------------------------------------------------")

        # Save the last model
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_latest.pth")) 

    total_finish_time = (time.time() - total_start_time)  # seconds
    logger.info('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))
    cleanup()

if __name__ == '__main__':
    ## Set Seeds
    torch.backends.cudnn.benchmark = True
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    
    ## Load yaml configuration file
    with open('training.yaml', 'r') as config:
        opt = yaml.safe_load(config)
    Train = opt['TRAINING']
    OPT = opt['OPTIM']

    main(opt, Train, OPT)