"""
Train script for the DMCE scheme in paper "Diffusion-Based Generative Prior for Low-Complexity MIMO Channel Estimation"
To run this script, please first download the code of DMCE from https://github.com/benediktfesl/Diffusion_channel_est
Then, copy the DMME and modules directories into Run_DMCE

!!! Note:
To run this file, please edit the __init__ function of DiffusionModel in DMCE/diffusion_model.py and change "model: networks.CNN," to "model,".
"""


import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from DMCE import utils, DiffusionModel, Trainer, Tester, CNN
from Channel_datasets.data_loader import Channels
from DiT_model_for_DMCE import DiT_NN


def args_parser():
    parser = argparse.ArgumentParser('DMCE_estimator', add_help=False)
    parser.add_argument('--img_size', default=[64, 16], nargs='+', type=int, help='[Nr, Nt]')
    parser.add_argument('--dataset_name', default='CDL-C', type=str, help="Name of the training dataset")
    parser.add_argument('--data_channels', default=2, type=int, help="{Re, Im}")
    parser.add_argument('--is_angular', default=True, type=bool, help="Whether transform H into angular domain")
    parser.add_argument('--norm_channels', default='global', type=str, help="dataset normalization method")
    parser.add_argument('--sigma_data', default=1.0, type=float, help="std of channel image")
    parser.add_argument('--spacing_list', default=[0.5], nargs='+', type=int, help="the antenna spacing")
    parser.add_argument('--num_workers', default=0, type=int, help="for dataloader")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")

    parser.add_argument('--patch_size', default=[4, 4], nargs='+', type=int, help='patch size')
    parser.add_argument('--hidden_size', default=128, type=int, help='The hidden dimension of the model')
    parser.add_argument('--num_heads', default=8, type=int, help='The number of attention heads')
    parser.add_argument('--depth', default=2, type=int, help='The number of Transformer blocks')
    parser.add_argument('--mlp_ratio', default=4, type=int, help='The ratio of hidden dim of the MLP')
    parser.add_argument('--attn_drop', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_drop', type=float, default=0.0, help='Projection dropout rate')

    return parser

def main():
    args = args_parser().parse_args()
    args.num_pilots = int(args.img_size[1])  # Nt * ratio
    args.log_path = './DMCE_checkpoints/DMCE_DiT/%s_Nt%d_Nr%d_ULA' % (args.dataset_name, args.img_size[1], args.img_size[0])
    os.makedirs(args.log_path, exist_ok=True)

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # set data params
    mode = '2D'
    complex_data = True
    data_shape = tuple([args.data_channels, args.img_size[0], args.img_size[1]]) # [2, Nr, Nt]

    # set Diffusion model params
    num_timesteps = 100 #int(np.random.choice([100, 300, 500, 1_000, 2_000]))
    loss_type = 'l2'
    which_schedule = 'linear'

    max_snr_dB = 40
    beta_start = 1 - 10**(max_snr_dB/10) / (1 + 10**(max_snr_dB/10))
    if num_timesteps == 5:
        beta_end = 0.95  # -22.5dB
    elif num_timesteps == 10:
        beta_end = 0.7  # -22.5dB
    elif num_timesteps == 50:
        beta_end = 0.2  # -22.5dB
    elif num_timesteps == 100:
        beta_end = 0.1 # -22.5dB
    elif num_timesteps == 300:
        beta_end = 0.035  # -23dB
    elif num_timesteps == 500:
        beta_end = 0.02 #-22dB
    elif num_timesteps == 1_000:
        beta_end = 0.01 #-22dB
    elif num_timesteps == 10_000:
        beta_end = 0.001 #-24dB
    else:
        beta_end = 0.035
    objective = 'pred_noise'  # one of 'pred_noise' (L_n), 'pred_x_0' (L_h), 'pred_post_mean' (L_mu)
    loss_weighting = False # bool(np.random.choice([True, False]))
    clipping = False
    reverse_method = 'reverse_mean'  # either 'reverse_mean' or 'ground_truth'
    reverse_add_random = False  # True: PDF Sampling method | False: Reverse Mean Forwarding method

    # diffusion model parameter dictionary, which is saved in 'sim_params.json'
    diff_model_dict = {
        'data_shape': data_shape,
        'complex_data': complex_data,
        'loss_type': loss_type,
        'which_schedule': which_schedule,
        'num_timesteps': num_timesteps,
        'beta_start': beta_start,
        'beta_end': beta_end,
        'objective': objective,
        'loss_weighting': loss_weighting,
        'clipping': clipping,
        'reverse_method': reverse_method,
        'reverse_add_random': reverse_add_random
    }

    kernel_size = (3, 3)
    n_layers_pre = 2
    max_filter = 64
    ch_layers_pre = np.linspace(start=1, stop=max_filter, num=n_layers_pre+1, dtype=int)
    ch_layers_pre[0] = 2
    ch_layers_pre = tuple(ch_layers_pre)
    ch_layers_pre = tuple(int(x) for x in ch_layers_pre)
    n_layers_post = 3
    ch_layers_post = np.linspace(start=1, stop=max_filter, num=n_layers_post+1, dtype=int)
    ch_layers_post[0] = 2
    ch_layers_post = ch_layers_post[::-1]
    ch_layers_post = tuple(ch_layers_post)
    ch_layers_post = tuple(int(x) for x in ch_layers_post)
    n_layers_time = 1
    ch_init_time = 16
    batch_norm = False
    downsamp_fac = 1

    # batch_norm = True
    cnn_dict = {
        'data_shape': data_shape,
        'n_layers_pre': n_layers_pre,
        'n_layers_post': n_layers_post,
        'ch_layers_pre': ch_layers_pre,
        'ch_layers_post': ch_layers_post,
        'n_layers_time': n_layers_time,
        'ch_init_time': ch_init_time,
        'kernel_size': kernel_size,
        'mode': mode,
        'batch_norm': batch_norm,
        'downsamp_fac': downsamp_fac,
        'device': device,
    }

    # set Trainer params
    batch_size = 128
    lr_init = 1e-4
    lr_step_multiplier = 1.0
    epochs_until_lr_step = 150
    num_epochs = 500
    val_every_n_batches = 2000
    num_min_epochs = 50
    num_epochs_no_improve = 20
    track_val_loss = True
    track_fid_score = False
    track_mmd = False
    use_fixed_gen_noise = True
    use_ray = False
    save_mode = 'best' # newest, all
    # dir_result = path.join(cwd, 'results')
    # timestamp = utils.get_timestamp()
    # dir_result = path.join(dir_result, timestamp)

    # Trainer parameter dictionary, which is saved in 'sim_params.json'
    trainer_dict = {
        'batch_size': batch_size,
        'lr_init': lr_init,
        'lr_step_multiplier': lr_step_multiplier,
        'epochs_until_lr_step': epochs_until_lr_step,
        'num_epochs': num_epochs,
        'val_every_n_batches': val_every_n_batches,
        'track_val_loss': track_val_loss,
        'track_fid_score': track_fid_score,
        'track_mmd': track_mmd,
        'use_fixed_gen_noise': use_fixed_gen_noise,
        'save_mode': save_mode,
        'mode': mode,
        # 'dir_result': str(dir_result),
        'use_ray': use_ray,
        'complex_data': complex_data,
        'num_min_epochs': num_min_epochs,
        'num_epochs_no_improve': num_epochs_no_improve,
        # 'fft_pre': fft_pre,
    }

    sim_dict = {
        'diff_model_dict': diff_model_dict,
        'cnn_dict': cnn_dict,
        'trainer_dict': trainer_dict,
        # 'tester_dict': tester_dict,
        # 'misc_dict': misc_dict
    }

    # ========================================================================
    # 1. Get datasets and loaders for channels
    # ========================================================================
    dataset = Channels(args, is_train=True)
    dataloader = DataLoader(dataset, batch_size=trainer_dict['batch_size'], shuffle=True, num_workers=args.num_workers, drop_last=True)

    # instantiate CNN, DiffusionModel, Trainer and Tester
    # create the denoiser:
    net = DiT_NN(img_size=args.img_size,
              in_channels=args.data_channels,
              patch_size=args.patch_size,
              hidden_size=args.hidden_size,
              num_heads=args.num_heads,
              depth=args.depth,
              mlp_ratio=args.mlp_ratio,
              attn_drop=args.attn_drop,
              proj_drop=args.proj_drop,
                 device=device)

    net = net.to(device)

    # cnn = CNN(**cnn_dict)
    diffusion_model = DiffusionModel(net, **diff_model_dict)

    print(f'Number of trainable model parameters: {diffusion_model.num_parameters}')

    # optimizer:
    # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_init)
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=lr_init)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs_until_lr_step, gamma=lr_step_multiplier, verbose=True)

    # trainer = Trainer(diffusion_model, data_train, data_val, **trainer_dict)
    # train_dict = trainer.train()
    # --------------- training -------------------
    train_loss = []
    for epoch in range(trainer_dict['num_epochs']):
        print('The current epoch:', epoch)
        train_losses_epochs = []
        for i, sample in enumerate(dataloader):
            H_data = sample['H_real_norm'].to(device)
            if args.is_angular:
                H_data = sample['H_real_norm_ang'].to(device)

            loss = diffusion_model(H_data)
            train_losses_epochs.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss.append(np.mean(train_losses_epochs))
        lr_scheduler.step()

    # save model parameters:
    model_save_name = 'DMCE' + 'ep' + str(trainer_dict['num_epochs']) + '.pt'
    torch.save({'model_state': diffusion_model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'config': args,
                'diff_model_dict': diff_model_dict,
                'cnn_dict': cnn_dict,
                'trainer_dict': trainer_dict,
                'train_loss': train_loss},
               os.path.join(args.log_path, model_save_name))

    # ------ plot the loss ------
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(train_loss)
    plt.xlabel("eopch")
    plt.ylabel("loss")
    plt.show()


if __name__ == '__main__':
    main()
