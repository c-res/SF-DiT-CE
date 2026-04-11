import torch
import hdf5storage
from torch.utils.data import Dataset
import numpy as np
import os

def spatial_angular_transform(H, inverse=False):
    if not inverse:
        # spatial -> angular
        H = torch.fft.ifft(H, dim=-2, norm="ortho")
        H = torch.fft.fft(H, dim=-1, norm="ortho")
    else:
        # angular -> spatial
        H = torch.fft.ifft(H, dim=-1, norm="ortho")
        H = torch.fft.fft(H, dim=-2, norm="ortho")
    return H


class Channels(Dataset):
    def __init__(self, args, is_train):
        spacings_list = args.spacing_list
        dataset_name = args.dataset_name
        norm = args.norm_channels
        channels_spatial = []
        self.spacings = np.copy(spacings_list)
        self.filenames = []
        self.num_pilots = args.num_pilots
        for spacing in spacings_list:
            current_path = os.getcwd()
            parent_dir = os.path.dirname(current_path)
            filename = parent_dir + '/Channel_datasets/%s_Nt%d_Nr%d_ULA%.2f_train.mat' % (dataset_name, args.img_size[1], args.img_size[0], spacing)
            if not is_train:
                filename = parent_dir + '/Channel_datasets/%s_Nt%d_Nr%d_ULA%.2f_test.mat' % (dataset_name, args.img_size[1], args.img_size[0], spacing)
            self.filenames.append(filename)
            contents = hdf5storage.loadmat(filename)
            channels = np.asarray(contents['output_h'], dtype=np.complex64)
            channels_spatial.append(channels[:, 0])

        channels_spatial = np.asarray(channels_spatial)
        channels_spatial = np.reshape(channels_spatial, (-1, channels_spatial.shape[-2], channels_spatial.shape[-1]))

        self.channels = channels_spatial.astype(np.complex64)
        Re_channel = np.real(self.channels)
        Im_channel = np.imag(self.channels)
        if type(norm) == list:
            self.mean_real = norm[0]
            self.mean_im = norm[1]
            self.std_real = norm[2]
            self.std_im = norm[3]
        elif norm == 'global':
            self.mean_real = Re_channel.mean()
            self.std_real = Re_channel.std()
            self.mean_im = Im_channel.mean()
            self.std_im = Im_channel.std()
        elif norm == 'no_norm':
            self.mean_real = 0.0
            self.std_real = 1.0
            self.mean_im = 0.0
            self.std_im = 1.0
        else:
            raise ValueError(f"Unknown norm mode: {norm}")
        self.std_real = np.maximum(self.std_real, 1e-8)
        self.std_im = np.maximum(self.std_im, 1e-8)

        N_t = args.img_size[1]
        k = np.arange(N_t).reshape(-1, 1)
        n = np.arange(N_t).reshape(1, -1)
        omega = np.exp(-2j * np.pi / N_t)
        W = (omega ** (k * n)) / np.sqrt(N_t)
        self.DFT = W

    def __len__(self):
        return len(self.channels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        H_cplx = self.channels[idx]
        Re = (H_cplx.real - self.mean_real) / self.std_real
        Im = (H_cplx.imag - self.mean_im) / self.std_im
        H_cplx_norm = (Re + 1j * Im).astype(np.complex64)
        H_real_norm = np.stack((H_cplx_norm.real, H_cplx_norm.imag), axis=0).astype(np.float32)
        H_cplx_norm1 = torch.from_numpy(H_cplx_norm.astype(np.complex64))
        H_cplx_norm_ang = spatial_angular_transform(H_cplx_norm1, inverse=False)
        H_cplx_norm_ang = H_cplx_norm_ang.numpy().astype(np.complex64)
        H_real_norm_ang = np.stack((H_cplx_norm_ang.real, H_cplx_norm_ang.imag), axis=0).astype(np.float32)
        P = self.DFT[:, :self.num_pilots].astype(np.complex64)
        sample = {'H_cplx_norm': H_cplx_norm,
                  'H_real_norm': H_real_norm,
                  'H_cplx_norm_ang': H_cplx_norm_ang,
                  'H_real_norm_ang': H_real_norm_ang,
                  'P_cplx': P,
                  }
        return sample
