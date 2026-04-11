import argparse
# from dotmap import DotMap
import copy
import torch
from tqdm import tqdm as tqdm
import math
import sys
import os
import numpy as np
from torch.utils.data import DataLoader

from Channel_datasets.data_loader import Channels

from DiT_Models.VP_DiT_denoiser import VP_DiTDenoiser
from DiT_Models.DiT_utils import add_weight_decay


def args_parser():
    parser = argparse.ArgumentParser('DiT_estimator', add_help=False)
    parser.add_argument('--img_size', default=[64, 16], nargs='+', type=int, help='[Nr, Nt]')
    parser.add_argument('--patch_size', default=[4, 4], nargs='+', type=int, help='patch size')
    parser.add_argument('--hidden_size', default=128, type=int, help='The hidden dimension of the model')
    parser.add_argument('--num_heads', default=8, type=int, help='The number of attention heads')
    parser.add_argument('--depth', default=2, type=int, help='The number of Transformer blocks')
    parser.add_argument('--mlp_ratio', default=4, type=int, help='The ratio of hidden dim of the MLP')
    parser.add_argument('--attn_drop', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_drop', type=float, default=0.0, help='Projection dropout rate')
    parser.add_argument('--dataset_name', default='CDL-C', type=str, help="Name of the training dataset")
    parser.add_argument('--data_channels', default=2, type=int, help="{Re, Im}")
    parser.add_argument('--is_angular', default=True, type=bool, help="Whether transform H into angular domain")
    parser.add_argument('--norm_channels', default='global', type=str, help="dataset normalization method")
    parser.add_argument('--spacing_list', default=[0.5], nargs='+', help="the antenna spacing")
    parser.add_argument('--num_workers', default=0, type=int, help="for dataloader")
    parser.add_argument('--batch_size', default=128, type=int, help="the training batch size")
    parser.add_argument('--training_epochs', default=1000, type=int, help="the antenna spacing")
    parser.add_argument('--start_epoch', default=0, type=int, help="the strat epoch for training")
    parser.add_argument('--weight_decay', default=0.000, type=float, help="weight decay of optimizer")
    parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
    parser.add_argument('--P_mean', default=1.5, type=float, help='Hyperparameter for noise schedule')
    parser.add_argument('--P_std', default=0.96, type=float, help='Hyperparameter for noise schedule')
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--ema_decay1', type=float, default=0.9999, help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996, help='The second ema to track')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")

    return parser


def Train():
    args = args_parser().parse_args()
    args.num_pilots = int(args.img_size[1])
    args.log_path = './models/VP_DiT_angular_D%d_H%d_head%d/%s_Nt%d_Nr%d_ULA' % (args.depth, args.hidden_size, args.num_heads,
                                                                         args.dataset_name, args.img_size[1],
                                                                         args.img_size[0])

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    os.makedirs(args.log_path, exist_ok=True)
    # ========================================================================
    # 1. Get datasets and loaders for channels
    # ========================================================================
    dataset = Channels(args, is_train=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # ========================================================================
    # 2. Create the model:
    # ========================================================================
    model = VP_DiTDenoiser(args)
    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))
    model.to(device)

    param_groups = add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    model.init_ema()

    # ========================================================================
    # 3. Train the model:
    # ========================================================================
    train_loss, val_loss = [], []
    for epoch in range(args.start_epoch, args.training_epochs):
        print('The current epoch:', epoch)
        model.train(True)
        train_loss_epoch = []
        for i, sample in enumerate(dataloader):
            H_data = sample['H_real_norm'].to(device)
            if args.is_angular:
                H_data = sample['H_real_norm_ang'].to(device)
            loss = model(H_data)
            loss_value = loss.item()
            train_loss_epoch.append(loss_value)
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_ema()

        train_loss.append(np.mean(train_loss_epoch))

        save_freq = 100
        if (epoch + 1) % save_freq == 0 or epoch + 1 == args.training_epochs:
            model_save_name = 'VP_DiT' + 'ep' + str(epoch + 1) + '.pt'
            torch.save({'model_state': model.state_dict(),
                        'ema_params1': model.ema_params1,
                        'ema_params2': model.ema_params2,
                        'optim_state': optimizer.state_dict(),
                        'config': args,
                        'train_loss': train_loss},
                       os.path.join(args.log_path, model_save_name))

if __name__ == '__main__':
    Train()