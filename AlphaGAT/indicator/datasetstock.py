import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

class Dataset_Stock(Dataset):
    def __init__(self, dataset,day_windows=30, flag='train', device ="cpu") -> None:
        assert flag in ['train', 'valid', 'test']
        assert dataset in ['BTC', 'CSI100', 'DIJA', 'HSI']
        self.flag = flag
        self.day_windows = day_windows
        self.device =device
        self.dataset = dataset
        self.VOLUME_SCALE = 1e10
        INDICATORS = [
            "close",
            "open",
            "high",
            "low",
            'vwap',
            'turn',
            'chg',
            'volume'
        ]

        data_dir = f"../../data/{self.dataset}/{flag}.csv"
        self.indicator_num = len(INDICATORS)
        self.data_np = convert_dataframe2numpy(data_dir, INDICATORS)
        self.VOLUME_SCALE = 1e9


    def __len__(self):
        return len(self.data_np) - self.day_windows

    def __getitem__(self, index:int):
        ###[date_windows, stock_num, indicator]
        data = self.data_np[index:index+self.day_windows].copy()

        next_data_close = self.data_np[index+self.day_windows, :, 0].copy()
        "Normalization allows all data to be normalized along the corresponding dimensions."

        data_close = data[-1,:,0]

        reward_ratio = (next_data_close - data_close) / data_close
        data[:,:,-1] = data[:,:,-1] / self.VOLUME_SCALE
        data = torch.as_tensor(data, dtype=torch.float32).to(self.device)
        reward_ratio = torch.as_tensor(reward_ratio, dtype=torch.float32).reshape(1, -1).to(self.device)

        ###Convert to percentage increase.
        reward_ratio = reward_ratio * 100
        return data, reward_ratio


def convert_dataframe2numpy(data_path, indicator):
    data_df = pd.read_csv(data_path)
    data_df = data_df.set_index(data_df.columns[0])
    if 'vwap' in indicator:
        data_df.loc[data_df['volume'] == 0, 'vwap'] = data_df.loc[data_df['volume'] == 0, 'close']
    nan_exists = data_df.isna().any().any()
    # Check for missing values.
    empty_exists = (data_df.applymap(lambda x: isinstance(x, (int, float)) and x == '')).any().any()
    assert (not nan_exists) and (not empty_exists)
    stock_num = len(data_df.Name.unique())
    array = data_df[indicator].to_numpy().reshape(-1, stock_num,
                                                  len(indicator))  ###[date, stock_num, indicator]
    return array

