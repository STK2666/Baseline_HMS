from tqdm import trange,tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger
from modules.model import GeneratorFullModel
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from frames_dataset import DatasetRepeater
import math
import os
from logger import Logger, Visualizer
import numpy as np
import imageio
from pytorch_msssim import ssim
import lpips
import wandb

def train_test(config, inpainting_network, bg_predictor, dense_motion_network, checkpoint, log_dir, dataset, test_dataset):
    wandb.init(
        # set the wandb project where this run will be logged
        project = config['dataset'],
        name = config['run_name'],

        # track hyperparameters and run metadata
        config={
            "learning_rate": config['train_params']['lr_generator'],
            "architecture": "MRAA",
            "repeats": config['train_params']['num_repeats'],
        }
    )

    train_params = config['train_params']
    optimizer = torch.optim.Adam(
        [{'params': list(inpainting_network.parameters()) +
                    list(dense_motion_network.parameters()), 'initial_lr': train_params['lr_generator']}],lr=train_params['lr_generator'], betas=(0.5, 0.999), weight_decay = 1e-4)

    optimizer_bg_predictor = None
    if bg_predictor:
        optimizer_bg_predictor = torch.optim.Adam(
            [{'params':bg_predictor.parameters(),'initial_lr': train_params['lr_generator']}],
            lr=train_params['lr_generator'], betas=(0.5, 0.999), weight_decay = 1e-4)

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(
            checkpoint, inpainting_network = inpainting_network, dense_motion_network = dense_motion_network, bg_predictor = bg_predictor,
            optimizer = optimizer, optimizer_bg_predictor = optimizer_bg_predictor)
        print('load success:', start_epoch)
        start_epoch += 1
    else:
        start_epoch = 0

    # def lambda_rule(epoch):
    #     lr_l = 1.0 - max(0, epoch - 49) / float(50)
    #     return lr_l
    # scheduler_optimizer = LambdaLR(optimizer, lr_lambda=lambda_rule)
    
    scheduler_optimizer = MultiStepLR(optimizer, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    if bg_predictor:
        scheduler_bg_predictor = MultiStepLR(optimizer_bg_predictor, train_params['epoch_milestones'],
                                              gamma=0.1, last_epoch=start_epoch - 1)
        # scheduler_bg_predictor = LambdaLR(optimizer, lr_lambda=lambda_rule)

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True,
                            num_workers=train_params['dataloader_workers'], drop_last=True)

    generator_full = GeneratorFullModel(bg_predictor, dense_motion_network, inpainting_network, train_params)

    if torch.cuda.is_available():
        generator_full = torch.nn.DataParallel(generator_full).cuda()

    bg_start = train_params['bg_start']

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'],
                checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            losses = {}

            inpainting_network.train()
            dense_motion_network.train()
            if bg_predictor:
                bg_predictor.train()

            for x in tqdm(dataloader):
                if(torch.cuda.is_available()):
                    x['driving'] = x['driving'].cuda()
                    x['source'] = x['source'].cuda()
                    x['driving_rdr'] = x['driving_rdr'].cuda()
                    x['source_rdr'] = x['source_rdr'].cuda()
                    x['driving_smpl'] = x['driving_smpl'].cuda()
                    x['source_smpl'] = x['source_smpl'].cuda()

                losses_generator, generated = generator_full(x)
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)
                loss.backward()

                clip_grad_norm_(dense_motion_network.parameters(), max_norm=10, norm_type = math.inf)
                if bg_predictor and epoch>=bg_start:
                    clip_grad_norm_(bg_predictor.parameters(), max_norm=10, norm_type = math.inf)

                optimizer.step()
                optimizer.zero_grad()
                if bg_predictor and epoch>=bg_start:
                    optimizer_bg_predictor.step()
                    optimizer_bg_predictor.zero_grad()

                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

            scheduler_optimizer.step()
            if bg_predictor:
                scheduler_bg_predictor.step()

            model_save = {
                'inpainting_network': inpainting_network,
                'dense_motion_network': dense_motion_network,
                'optimizer': optimizer,
            }
            if bg_predictor and epoch>=bg_start:
                model_save['bg_predictor'] = bg_predictor
                model_save['optimizer_bg_predictor'] = optimizer_bg_predictor

            logger.log_epoch(epoch, model_save, inp=x, out=generated)

            # metric
            with torch.no_grad():
                l1_metric = torch.abs(x['driving'] - generated['prediction']).mean().cpu().numpy()
                losses['L1'] = l1_metric
                ssim_metric = ssim(x['driving'], generated['prediction'], data_range=1).mean().cpu().numpy()
                losses['SSIM'] = ssim_metric
                mse = torch.mean((x['driving'] - generated['prediction']) ** 2)
                if mse == 0:
                    psnr_metric = float('inf')
                else:
                    psnr_metric = 20 * torch.log10(1.0 / torch.sqrt(mse)).cpu().numpy()
                losses['PSNR'] = psnr_metric

            wandb.log(losses)
            del x
            torch.cuda.empty_cache()
    wandb.finish()

