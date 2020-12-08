import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class BtpDataset(Dataset):
    """Btp time series dataset."""

    def __init__(self, csv_file, normalize=True):
        """
        Args:
            csv_file (string): path to csv file
            normalize (bool): whether to normalize the data in [-1,1]
        """
        df = pd.read_csv(csv_file, sep=";")
        df['Timestamp'] = pd.to_datetime(df["data_column"].map(str) + " " + df["orario_column"], dayfirst=True)
        df = df.drop(['data_column', 'orario_column'], axis=1).set_index("Timestamp")
        btp_price = df.BTP_Price
        data = torch.from_numpy(
            np.expand_dims(np.array([group[1] for group in btp_price.groupby(df.index.date)]), -1)).float()
        self.data = self.normalize(data) if normalize else data
        self.seq_len = data.size(1)

        # Estimates distribution parameters of deltas (Gaussian) from normalized data
        original_deltas = data[:, -1] - data[:, 0]
        self.original_deltas = original_deltas
        self.or_delta_max, self.or_delta_min = original_deltas.max(), original_deltas.min()
        deltas = self.data[:, -1] - self.data[:, 0]
        self.deltas = deltas
        self.delta_mean, self.delta_std = deltas.mean(), deltas.std()
        self.delta_max, self.delta_min = deltas.max(), deltas.min()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def normalize(self, x):
        """Normalize input in [-1,1] range, saving statics for denormalization"""
        self.max = x.max()
        self.min = x.min()
        return (2 * (x - x.min()) / (x.max() - x.min()) - 1)

    def denormalize(self, x):
        """Revert [-1,1] normalization"""
        if not hasattr(self, 'max') or not hasattr(self, 'min'):
            raise Exception("You are calling denormalize, but the input was not normalized")
        return 0.5 * (x * self.max - x * self.min + self.max + self.min)

    def sample_deltas(self, number):
        """Sample a vector of (number) deltas from the fitted Gaussian"""
        return (torch.randn(number, 1) + self.delta_mean) * self.delta_std

    def normalize_deltas(self, x):
        return ((self.delta_max - self.delta_min) * (x - self.or_delta_min) / (
                    self.or_delta_max - self.or_delta_min) + self.delta_min)


class ETHDataset(Dataset):
    """ETH Dataset."""

    def __init__(
            self,
            csv_file,
            first_data_point_date,
            last_data_point_date,
            condition_size,
            dst_col='closePrice',
            normalize=True
    ):
        """
        :param csv_file(string): Path to the csv file.
        :param transform(callable, optional): Optional transform to be applied
            on a sample.
        :param first_data_point_date(string): Date of first datapoint that
            will be used.
        :param last_data_point_date(string): Date of last datapoint that
            will be used.
        :param condition_size(int): Size of condition vector.
        :param dst_col(string): Column that should be used as main data
            e.g. close_price.
        """

        self.eth_df = pd.read_csv(csv_file)
        # self.normalize = normalize
        self.all_data = list(self.eth_df[dst_col])
        for idx, val in enumerate(self.eth_df['Date']):
            if val == first_data_point_date:
                first_data_point_idx = idx
                break
        for idx, val in enumerate(self.eth_df['Date']):
            if val == last_data_point_date:
                last_data_point_idx = idx
                break
        self.first_data_point_idx = first_data_point_idx
        self.last_data_point_idx = last_data_point_idx
        self.main_data = list(self.eth_df[dst_col])
        self.main_data = np.array(self.main_data[first_data_point_idx:last_data_point_idx])
        # self.max = max(self.main_data)
        # self.min = min(self.main_data)
        num_seq = int(len(self.main_data) / 48)
        print(num_seq)
        self.main_data = self.main_data[len(self.main_data) - num_seq * 48:]
        self.main_data = self.main_data.reshape(num_seq, 48)
        self.main_data = self.main_data.reshape(num_seq, 1, 48)
        self.main_data = torch.from_numpy(self.main_data)
        # self.main_data = self.main_data.type(torch.DoubleTensor)
        self.main_data = self.normalize(self.main_data) if normalize else self.main_data
        self.seq_len = self.main_data.size(1)

        self.condition_size = condition_size
        self.dst_col = dst_col

    def __len__(self):
        return len(self.main_data)

    def __getitem__(self, idx):
        # print(self.main_data[idx].size())
        return self.main_data[idx]

    def normalize(self, x):
        """Normalize input in [-1,1] range, saving statics for denormalization"""
        self.max = x.max()
        self.min = x.min()
        return (2 * (x - x.min()) / (x.max() - x.min()) - 1)

    def denormalize(self, input):
        """Revert [-1,1] normalization"""
        if not hasattr(self, 'max') or not hasattr(self, 'min'):
            raise Exception("You are calling denormalize, but the input was not normalized")
        return 0.5 * (input * self.max - input * self.min + self.max + self.min)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors"""

    def __call__(self, sample):
        sample = np.array([sample])
        return torch.from_numpy(sample)