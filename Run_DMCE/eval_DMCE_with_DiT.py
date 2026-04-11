"""
Train and test script for the DMCE with DiT model.
!!! Note:
To run this file, please edit the __init__ function of DiffusionModel in DMCE/diffusion_model.py and change "model: networks.CNN," to "model,".
"""
import os
import argparse
import numpy as np
import torch
import itertools
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm as tqdm

import DMCE

from Channel_datasets.data_loader import Channels, spatial_angular_transform


from DiT_model_for_DMCE import DiT_NN

def args_parser():
    parser = argparse.ArgumentParser()

    # hardware arguments
    parser.add_argument('--gpu', type=int, default=0)
    # dataset arguments
    parser.add_argument('--train_dataset', type=str, default='CDL-C', help="The training dataset")
    parser.add_argument('--test_dataset', type=str, default='CDL-C', help="The test dataset")
    parser.add_argument('--img_size', default=[64, 16], nargs='+', type=int, help='[Nr, Nt]')
    parser.add_argument('--spacing', nargs='+', type=float, default=[0.5])
    parser.add_argument('--pilot_alpha', nargs='+', type=float, default=[1.0])  # alpha = N_p / N_t

    args = parser.parse_args()
    return args


def main():
    args = args_parser()
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    # ========================================================================
    # 1. read and load the model:
    # ========================================================================
    current_path = os.getcwd()
    # parent_dir = os.path.dirname(current_path)
    target_dir = current_path + '/DMCE_checkpoints/DMCE_DiT/%s_Nt%d_Nr%d_ULA' % (args.train_dataset, args.img_size[1], args.img_size[0])


    # load files:
    target_file = os.path.join(target_dir, 'DMCEep500.pt')
    contents = torch.load(target_file, map_location=device)
    training_args = contents['config']
    cnn_dict = contents['cnn_dict']
    diff_model_dict = contents['diff_model_dict']

    return_all_timesteps = False # evaluates all intermediate MSEs
    # fft_pre = True # learn channel distribution in angular domain through Fourier transform
    reverse_add_random = False # re-sampling in the reverse process

    cnn_dict['device'] = device

    # instantiate the neural network
    # cnn = DMCE.CNN(**cnn_dict)

    net = DiT_NN(img_size=training_args.img_size,
                 in_channels=training_args.data_channels,
                 patch_size=training_args.patch_size,
                 hidden_size=training_args.hidden_size,
                 num_heads=training_args.num_heads,
                 depth=training_args.depth,
                 mlp_ratio=training_args.mlp_ratio,
                 attn_drop=training_args.attn_drop,
                 proj_drop=training_args.proj_drop,
                 device=device)

    net = net.to(device)

    # instantiate the diffusion model and give it a reference to the unet model
    diffusion_model = DMCE.DiffusionModel(net, **diff_model_dict)

    # load the parameters of the pre-trained model into the DiffusionModel instance
    diffusion_model.load_state_dict(contents['model_state'])
    diffusion_model.reverse_add_random = reverse_add_random

    # ========================================================================
    # 2. prepare the test dataset
    # ========================================================================
    training_dataset = Channels(training_args, is_train=False)
    snr_range = np.arange(-10, 32.5, 2.5)
    spacing_range = np.asarray(args.spacing)
    pilot_alpha_range = np.asarray(args.pilot_alpha)
    noise_range = 10 ** (-snr_range / 10.)
    # Number of validation channels
    num_test_samples = 1000

    nmse_log = np.zeros((len(spacing_range), len(pilot_alpha_range), len(snr_range), num_test_samples))
    meta_params = itertools.product(spacing_range, pilot_alpha_range)

    for meta_idx, (spacing, pilot_alpha) in tqdm(enumerate(meta_params)):
        spacing_idx, pilot_alpha_idx = np.unravel_index(meta_idx, (len(spacing_range), len(pilot_alpha_range)))
        val_args = copy.deepcopy(training_args)
        val_args.dataset_name = args.test_dataset
        val_args.norm_channels = [training_dataset.mean_real, training_dataset.mean_im, training_dataset.std_real,
                                  training_dataset.std_im]
        val_args.spacing_list = [spacing]
        val_args.num_pilots = int(np.floor(training_args.img_size[1] * pilot_alpha))

        val_dataset = Channels(val_args, is_train=False)
        #
        test_batch_size = 20
        val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0, drop_last=True)
        for batch_idx, val_sample in enumerate(val_loader):
            sample_idx_start = batch_idx * test_batch_size
            P = val_sample['P_cplx'].to(device)
            P_H = torch.conj(torch.transpose(P, -1, -2))

            val_H_real = val_sample['H_real_norm'].to(device)
            val_H = val_H_real[:, 0] + 1j * val_H_real[:, 1]

            for snr_idx, local_noise in enumerate(noise_range):
                print('SNR_idx=', snr_idx)
                HP = torch.matmul(val_H, P)
                N = np.sqrt(local_noise) * torch.randn_like(HP)
                Y = HP + N

                # compute the LS estimate:
                H_LS = torch.matmul(Y, P_H)
                H_LS_real = torch.view_as_real(H_LS).permute(0, 3, 1, 2).contiguous()

                # normalize the LS estimation:
                normalize_coff = torch.tensor(1 / np.sqrt(local_noise + 1), device=device, dtype=H_LS_real.dtype)
                H_LS_real = H_LS_real * normalize_coff

                # convert to angular domain:
                H_LS_cplx = torch.view_as_complex(H_LS_real.permute((0, 2, 3, 1)).contiguous())
                if training_args.is_angular:
                    H_LS_cplx = spatial_angular_transform(H_LS_cplx, inverse=False)

                H_LS_real = torch.view_as_real(H_LS_cplx).permute(0, 3, 1, 2).contiguous()
                SNR = 1 / (local_noise)
                # estimate t_hat, the time step that corresponds to the correct SNR
                snr_diffusion = diffusion_model.snrs
                t = int(torch.abs(snr_diffusion - SNR).argmin())

                H_pred = diffusion_model.reverse_sample_loop(H_LS_real, t, return_all_timesteps=return_all_timesteps)
                H_pred_cplx = torch.view_as_complex(H_pred.permute((0, 2, 3, 1)).contiguous())

                if training_args.is_angular:
                    H_pred_cplx = spatial_angular_transform(H_pred_cplx, inverse=True)

                # compute the NMSE metric:
                nmse_log_temp = torch.sum(torch.square(torch.abs(H_pred_cplx - val_H)), dim=(-1, -2)) / torch.sum(
                    torch.square(torch.abs(val_H)), dim=(-1, -2))
                nmse_log_temp = nmse_log_temp.detach().cpu().numpy()

                if not np.isfinite(nmse_log_temp).all():
                    print("NaN/Inf at snr_idx=", snr_idx, "snr=", snr_range[snr_idx])

                nmse_log[spacing_idx, pilot_alpha_idx, snr_idx,
                sample_idx_start:(sample_idx_start + test_batch_size)] = nmse_log_temp

    #
    avg_nmse = np.mean(nmse_log, axis=-1)
    print(avg_nmse)
    print(10 * np.log10(avg_nmse))

    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 14
    plt.figure(figsize=(10, 10))
    for alpha_idx, local_alpha in enumerate(pilot_alpha_range):
        plt.plot(snr_range, 10 * np.log10(avg_nmse[0, alpha_idx]),
                 linewidth=4, label='Alpha=%.2f' % local_alpha)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()