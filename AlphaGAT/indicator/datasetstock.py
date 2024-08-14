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

        assert dataset in ['BTC', 'CSI100', 'DIJA', 'HSI']

        '''定义需要使用到的指标，可以自己添加或者选择'''

        if dataset =='BTC':
            INDICATORS = [
                "close",
                "open",
                "high",
                "low",
                'volume'
            ]
            self.VOLUME_SCALE = 1e10
        elif dataset=='CSI100':
            INDICATORS = [
                "close",
                "open",
                "high",
                "low",
                'vwap',
                'turn',
                'chg',
                'swing',
                'volume'
            ]
            self.VOLUME_SCALE = 1e8

        elif dataset=='HSI':
            INDICATORS = [
                "close",
                "open",
                "high",
                "low",
                'vwap',
                'turn',
                'chg',
                'swing',
                'volume'
            ]
            self.VOLUME_SCALE = 1e8

        elif dataset == 'DIJA':
            INDICATORS = [
                "close",
                "open",
                "high",
                "low",
                'turn',
                'chg',
                'volume'
            ]
            self.VOLUME_SCALE = 1e8



        else:
            raise ValueError("This dataset is not supported ")



        data_dir = f"../../data/{self.dataset}/{flag}.csv"

        self.indicator_num = len(INDICATORS)
        self.data_np = convert_dataframe2numpy(data_dir, INDICATORS)

        # self.VOLUME_SCALE = 1e9
        # self.SWING_SCALE = 10


    def __len__(self):
        return len(self.data_np) - self.day_windows

    def __getitem__(self, index:int):
        ###[date_windows, stock_num, indicator]
        data = self.data_np[index:index+self.day_windows].copy()

        next_data_close = self.data_np[index+self.day_windows, :, 0].copy()

        data_close = data[-1,:,0]

        reward_ratio = (next_data_close - data_close) / data_close

        # if self.dataset=='CSI100' or self.dataset=='HSI':
        #     data[:,:,4] = data[:,:,4] / data_close.reshape((1,-1,1))
        #
        # ####所有的价格都除以最后一天的收盘价,可以把数据价格统一归一化到1附近
        # data[:,:,:4] = data[:,:,:4] / data_close.reshape((1,-1,1))     ####np.repeat(last_close, self.indicator_num, axis=2)

        if self.dataset=='CSI100' or self.dataset=='HSI':
            data[:,:,:5] = data[:,:,:5] / data_close.reshape((1,-1,1))
        else:
            data[:, :, :4] = data[:, :, :4] / data_close.reshape((1, -1, 1))

        ####换手率不需要归一化，因为一般都在1附近左右  成交量需要归一化  除以统一的缩放因子即可
        data[:,:,-1] = data[:,:,-1] / self.VOLUME_SCALE
        # data[:,:,-2] = data[:,:,-2] / self.SWING_SCALE

        '''归一化处理  可以将所有数据在对应的维度进行归一化'''

        data = torch.as_tensor(data, dtype=torch.float32).to(self.device)
        reward_ratio = torch.as_tensor(reward_ratio, dtype=torch.float32).reshape(1, -1).to(self.device)

        ###转换为百分比的涨幅
        reward_ratio = reward_ratio * 100
        return data, reward_ratio


def convert_dataframe2numpy(data_path, indicator):
    data_df = pd.read_csv(data_path)
    data_df = data_df.set_index(data_df.columns[0])
    if 'vwap' in indicator:
        data_df.loc[data_df['volume'] == 0, 'vwap'] = data_df.loc[data_df['volume'] == 0, 'close']
    # data_df.loc[data_df['volume'] == 0, 'vwap'] = data_df.loc[data_df['volume'] == 0, 'close']
    # 检查是否存在 NaN
    nan_exists = data_df.isna().any().any()

    # 检查是否存在空白的数字
    empty_exists = (data_df.applymap(lambda x: isinstance(x, (int, float)) and x == '')).any().any()

    assert (not nan_exists) and (not empty_exists)


    stock_num = len(data_df.Name.unique())
    array = data_df[indicator].to_numpy().reshape(-1, stock_num,
                                                  len(indicator))  ###[date, stock_num, indicator]
    return array




# """test for dataset"""
# ds = Dateset_Stock(day_windows=30, flag='train', device="cuda:0")
# data, y = ds.__getitem__(0)
#
# print(data)
# print(y)