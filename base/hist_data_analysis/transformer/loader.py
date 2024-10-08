import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import logging
import utils
import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def include_time_repr(df, params, dts, cors, uniqs, args):
    """
    Add time representations to a DataFrame.
    :param df: dataframe
    :param dts: list of str
    :param cors: list of str
    :param uniqs: list of str
    :param args: list of arguments
    :return: dataframe
    """
    for i, dtime in enumerate(dts):
        timestamps = df['datetime']
        tr_cor = TimeRepr(timestamps, dtime, args=args[i][0])
        tr_uniq = TimeRepr(timestamps, dtime, args=args[i][1])

        df[f'COR_{dtime.upper()}'] = getattr(tr_cor, cors[i])
        df[f'UNIQ_{dtime.upper()}'] = getattr(tr_uniq, uniqs[i])

        params["t"].extend([f'COR_{dtime.upper()}', f'UNIQ_{dtime.upper()}'])

    return df, params


class TimeRepr():
    def __init__(self, timestamps, dtime, args):
        """
        Initializes a time representation class.
        :param timestamps: pandas series
        :param dtime: datetime attribute (e.g., 'day', 'month', 'date')
        :param args: list of arguments for the functions
        """
        self.timestamps = timestamps
        self.dtime = dtime
        self.args = args

    @property
    def sine(self):
        """
        Calculate sine representation of timestamps.
        :return: pandas series
        """
        period, _, shift = self.args
        self.timestamps = self.timestamps.dt.__getattribute__(self.dtime)

        sine_result = np.sin(np.pi*(self.timestamps-shift)/period)
        sine_result += sine_result.iloc[1]/10

        return sine_result

    @property
    def cosine(self):
        """
        Calculate cosine representation of timestamps.
        :return: pandas series
        """
        period, _, shift = self.args
        self.timestamps = self.timestamps.dt.__getattribute__(self.dtime)

        cosine_result = np.cos(np.pi*(self.timestamps-shift)/period)
        cosine_result += cosine_result.iloc[1]/10

        return cosine_result
    
    @property
    def sawtooth(self):
        """
        Calculate sawtooth representation of timestamps.
        :return: pandas series
        """
        period, _, shift = self.args
        self.timestamps = self.timestamps.dt.__getattribute__(self.dtime)

        sawtooth_result = (self.timestamps-shift)/period
        sawtooth_result += sawtooth_result.iloc[1]/10

        return sawtooth_result
    
    @property
    def cond_sawtooth(self):
        """
        Calculate conditional sawtooth representation of timestamps.
        :return: numpy array
        """
        period, total, shift = self.args
        self.timestamps = self.timestamps.dt.__getattribute__(self.dtime)

        cond_sawtooth_result = np.where(self.timestamps <= period, 
                                        (self.timestamps-shift)/period,
                                        (total-self.timestamps-shift)/period)
        cond_sawtooth_result += cond_sawtooth_result[1]/10

        return cond_sawtooth_result

    @property
    def linear(self):
        """
        Calculate linear representation of timestamps.
        
        :return: numpy array
        """
        total = (self.timestamps.iloc[-1] - self.timestamps.iloc[0]).total_seconds()
        line = list(map(lambda t: 1e-9 + (t - self.timestamps.iloc[0]).total_seconds()/total, self.timestamps))

        line = np.array(line) + line[1]/10
    
        return line


def load(path, time_repr, y, normalize=True):
    """
    Loads and preprocesses raw_data from a CSV file.

    :param path: path to the CSV file
    :param normalize: normalization flag
    :param time_repr: tuple
    :return: dataframe
    """

    def rename_columns(col):
        if col == 'datetime':
            return col
        elif col.endswith('.1'):
            return col.replace('.1', '_std')
        else:
            return col + '_mean'

    df = pd.read_csv(path, parse_dates=["datetime"], low_memory=False)
    df = df.drop(0)  # Drop the first column that consists of 'mean', 'std' strings
    # Apply the renaming function to get mean and std features
    df.columns = [rename_columns(col) for col in df.columns]

    df.sort_values(by='datetime', inplace=True)
    # Convert object columns to float, except 'datetime'
    df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').astype(float)

    '''params = {"X": [x for x in df.columns if x != 'datetime' and x != 'fuelVolumeFlowRate_std' and x != y],
              "t": []}'''

    params = {"X": ['inclinometer-raw_mean', 'inclinometer-raw_std', 'trueHeading_mean', 'trueHeading_std',
                    'windAngle_mean', 'windAngle_std', 'windSpeed_mean', 'windSpeed_std', 'longitudinalWaterSpeed_mean',
                    'longitudinalWaterSpeed_std', 'speedKmh_mean', 'speedKmh_std', 'speedKnots_mean', 'speedKnots_std'],
              "t": []}

    df, params = include_time_repr(df, params, *time_repr)

    if os.path.exists('./stats.json'):
        stats = utils.load_json(filename='./stats.json')
    else:
        stats = utils.get_stats(df, path='./')

    if normalize:
        df = utils.normalize(df, stats, exclude=['datetime', 'COR_MONTH', 'COR_DAY', 'COR_HOUR', 'COR_DATE',
                                                 'UNIQ_MONTH', 'UNIQ_DAY', 'UNIQ_HOUR', 'UNIQ_DATE', 'COR_SECOND',
                                                 'UNIQ_SECOND', y, 'fuelVolumeFlowRate_std'])

    # nan_counts = df.isna().sum() / len(df) * 100
    # logger.info("NaN counts for columns in X: %s", nan_counts)

    return df, params


class TSDataset(Dataset):
    def __init__(self, df, seq_len, X, t, y):
        """
        Initializes a time series dataset.

        :param df: dataframe
        :param seq_len: length of the input sequence
        :param X: input features names
        :param t: time-related features names
        :param y: target variables names
        """
        self.seq_len = seq_len

        y_nan = df[[y]].isna().any(axis=1)
        df.loc[y_nan, :] = float('nan')

        self.X = pd.concat([df[X], df[t]], axis=1)
        self.y = df[y]

    def __len__(self):
        """
        :return: number of sequences that can be created from dataset X
        """
        return self.X.shape[0] // self.seq_len - 1
        # return self.X.shape[0] - self.seq_len + 1 # -1
    
    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        :param idx: index of the sample
        :return: tuple containing input features sequence, target variables sequence and their respective masks
        """
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
    
        X, y = self.X.iloc[start_idx:end_idx].values, self.y.iloc[end_idx+1]

        mask_X, mask_y = pd.isnull(X).astype(int), int(pd.isnull(y))

        X, y = torch.FloatTensor(X), torch.FloatTensor([y])
        mask_X, mask_y = torch.FloatTensor(mask_X), torch.FloatTensor([mask_y])


        if mask_y == 1:

            y = torch.tensor([-1])
            X = X.fill_(-2)

            mask_y_1d = torch.zeros(1)
            mask_X_1d = torch.zeros(self.seq_len)
            # mask_y_1d = torch.ones(1)
        else:

            X = X.masked_fill(mask_X == 1, -2)

            mask_X_1d = torch.ones(self.seq_len)
            mask_y_1d = torch.ones(1)
            # mask_y_1d = torch.zeros(1)
            for i in range(self.seq_len):
                if torch.any(mask_X[i] == 1):
                    mask_X_1d[i] = 0
                    # mask_X_1d[i] = 1

        '''has_positive = torch.gt(y, 0).any()

        if has_positive:
            print(f'Positive value: {y}')'''

        return X, y, mask_X_1d, mask_y_1d


def split(dataset, vperc):
    """
    Splits a dataset into training and validation sets.

    :param dataset: dataset
    :param vperc: percentage of raw_data to allocate for validation
    :return: tuple containing training and validation datasets
    """
    ds_seqs = int(len(dataset))

    valid_seqs = int(vperc * ds_seqs)
    train_seqs = ds_seqs - valid_seqs

    return random_split(dataset, [train_seqs, valid_seqs])