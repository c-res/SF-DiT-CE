import argparse
import copy
import torch
from tqdm import tqdm as tqdm
import os
import numpy as np
import itertools
from torch.utils.data import DataLoader

from Channel_datasets.data_loader import Channels, spatial_angular_transform
from DiT_Models.VE_DiT_denoiser import VE_DiTDenoiser


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--train_dataset', type=str, default='CDL-C', help="The training dataset")
    parser.add_argument('--test_dataset', type=str, default='CDL-C', help="The test dataset")
    parser.add_argument('--img_size', default=[64, 16], nargs='+', type=int, help='[Nr, Nt]')
    parser.add_argument('--hidden_size', default=128, type=int, help='The hidden dimension of the model')
    parser.add_argument('--num_heads', default=8, type=int, help='The number of attention heads')
    parser.add_argument('--depth', default=2, type=int, help='The number of Transformer blocks')
    parser.add_argument('--spacing', nargs='+', type=float, default=[0.5])
    parser.add_argument('--pilot_alpha', nargs='+', type=float, default=[1.0])

    # change these settings to eval different variants of SF-DiT-CE
    parser.add_argument('--is_angular', default=True, type=bool, help="Whether transform H into angular domain")
    parser.add_argument('--predict_obj', default='X_predict',
                        choices=['X_predict', 'V_predict', 'epsilon_predict'],
                        help="The network prediction objective")
    parser.add_argument('--loss_type', default='V_loss', choices=['V_loss', 'X_loss', 'epsilon_loss'],
                        help="The loss type")

    args = parser.parse_args()
    return args


def Evaluate():
    args = args_parser()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    current_path = os.getcwd()

    save_path_name = 'VE_DiT'
    if args.is_angular:
        save_path_name = save_path_name + '_angular'
    else:
        save_path_name = save_path_name + '_spatial'

    save_path_name = save_path_name + '_' + args.predict_obj
    if args.predict_obj == 'X_predict':
        save_path_name = save_path_name + '_' + args.loss_type

    target_dir = (current_path + '/VE_checkpoints/' + save_path_name
                + '_D%d_H%d_head%d/%s_Nt%d_Nr%d_ULA' % (args.depth, args.hidden_size, args.num_heads,
                                                        args.train_dataset, args.img_size[1], args.img_size[0]))

    # load files:
    target_file = os.path.join(target_dir, 'VE_DiTep1000.pt')
    contents = torch.load(target_file, map_location=device)
    training_args = contents['config']
    model = VE_DiTDenoiser(training_args)
    model.to(device)
    model.load_state_dict(contents['model_state'])
    ema_state_dict1 = contents['ema_params1']
    ema_state_dict2 = contents['ema_params2']
    model.ema_params1 = ema_state_dict1
    model.ema_params2 = ema_state_dict2
    model.eval()
    model_ema = copy.deepcopy(model)
    with torch.no_grad():
        for name, p in model_ema.named_parameters():
            if name in ema_state_dict1:
                p.copy_(ema_state_dict1[name])
    model_ema.eval()

    training_dataset = Channels(training_args, is_train=False)
    snr_range = np.arange(-10, 32.5, 2.5)
    spacing_range = np.asarray(args.spacing)
    pilot_alpha_range = np.asarray(args.pilot_alpha)
    noise_range = 10 ** (-snr_range / 10.)
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

        test_batch_size = 100
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
                noise = torch.randn_like(HP.real) + 1j * torch.randn_like(HP.real)
                noise = np.sqrt(local_noise) * noise
                Y = HP + noise
                H_LS_cplx = torch.matmul(Y, P_H)
                bsz = H_LS_cplx.size(0)
                if args.is_angular:
                    H_LS_cplx = spatial_angular_transform(H_LS_cplx, inverse=False)
                H_LS_real = torch.view_as_real(H_LS_cplx).permute(0, 3, 1, 2).contiguous()
                sigma_y = torch.tensor(np.sqrt(local_noise), device=device, dtype=H_LS_real.dtype)
                sigma_y = sigma_y.view(1, 1, 1, 1).expand(bsz, 1, 1, 1)
                if args.predict_obj == 'X_predict':
                    H_pred, _ = model_ema.edm_sampling(H_LS_real, sigma_y)
                elif args.predict_obj == 'V_predict':
                    H_pred, _ = model_ema.x0_from_velocity(H_LS_real, sigma_y)
                elif args.predict_obj == 'epsilon_predict':
                    H_pred, _ = model_ema.x0_from_score(H_LS_real, sigma_y)
                else:
                    raise ValueError(f"Unknown Network Prediction Objective: {args.predict_obj}")

                H_pred_cplx = torch.view_as_complex(H_pred.permute((0, 2, 3, 1)).contiguous())
                if args.is_angular:
                    H_pred_cplx = spatial_angular_transform(H_pred_cplx, inverse=True)

                nmse_log_temp = torch.sum(torch.square(torch.abs(H_pred_cplx - val_H)), dim=(-1, -2)) / torch.sum(
                    torch.square(torch.abs(val_H)), dim=(-1, -2))
                nmse_log_temp = nmse_log_temp.detach().cpu().numpy()

                if not np.isfinite(nmse_log_temp).all():
                    print("NaN/Inf at snr_idx=", snr_idx, "snr=", snr_range[snr_idx])

                nmse_log[spacing_idx, pilot_alpha_idx, snr_idx,
                sample_idx_start:(sample_idx_start + test_batch_size)] = nmse_log_temp

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
    Evaluate()

