import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
import math
from collections import namedtuple


class MaskedMSELoss(nn.Module):
    """
    MaskedMSELoss that utilizes masks
    """
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, true, mask):
        pred = pred.squeeze()
        # Compute element-wise squared difference
        squared_diff = (pred - true) ** 2
        # Apply mask to ignore certain elements
        mask = mask.float()
        loss = squared_diff * mask
        # Compute the mean loss only over non-masked elements
        loss = loss.sum() / (mask.sum() + 1e-8)

        return loss


def get_max(arr):
    """
    Get the maximum value and its index from an array.

    :param arr: numpy array
    :return: namedtuple
    """
    Info = namedtuple('Info', ['value', 'index'])

    max_index = np.argmax(arr)
    max_value = arr[max_index]

    return Info(value=max_value, index=max_index)


def hot3D(t, k, device):
    """
    Encode 3D tensor into one-hot format.

    :param t: tensor of shape (dim_0, dim_1, dim_2)
    :param k: int number of classes
    :param device: device
    :return: tensor of shape (dim_0, dim_1, k)
    """
    dim_0, dim_1, _ = t.size()
    t_hot = torch.zeros(dim_0, dim_1, k, device=device)

    for x in range(dim_0):
        for y in range(dim_1):
            for z in t[x, y]:
                t_hot[x, y] = torch.tensor(one_hot(z.item(), k=k))

    return t_hot.to(device)


def one_hot(val, k):
    """
    Convert categorical value to one-hot encoded representation.

    :param val: float
    :param k: number of classes
    :return: list
    """

    if math.isnan(val):
        encoding = [val for _ in range(k)]
    else:
        encoding = [0 for _ in range(k)]
        encoding[int(val)] = 1

    return encoding


def get_path(dirs, name=""):
    """
    Get the path by joining directory names.
    :param dirs: list
    :param name: name of the path
    :return: the path
    """
    dir_path = os.path.join(*dirs)
    os.makedirs(dir_path, exist_ok=True)

    return os.path.join(dir_path, name)


def save_json(data, filename):
    """
    Save raw_data to a JSON file.
    :param data: dictionary
    :param filename: str
    """
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_json(filename):
    """
    Load raw_data from a JSON file.
    :param filename: str
    :return: dictionary
    """
    with open(filename, 'r') as f:
        data = json.load(f)

    return data


def save_csv(data, filename):
    """
    Save raw_data to a CSV file.

    :param data: dictionary
    :param filename: str
    """
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def visualize(vizualization, *args):

    if vizualization == 'losses':
        train_loss, val_loss = args
        epochs = [i+1 for i in range(len(train_loss))]

        # Plot the losses
        plt.plot(epochs, train_loss, 'r', label='Training Loss')
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
        # Add titles and labels
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        # Add a legend
        plt.legend()
        # Show the plot
        plt.show()

    elif vizualization == 'training_predictions':
        y_true, y_pred = args
        epochs = [i + 1 for i in range(len(y_true))]

        # Plot the losses
        plt.plot(epochs, y_true, 'o', label='True value')
        #plt.plot(epochs, y_pred, 'x', label='Predicted value')
        # Add titles and labels
        plt.title('True vs Predicted Value')
        plt.xlabel('Epochs')
        plt.ylabel('Predicted value')
        # Add a legend
        plt.legend()
        # Show the plot
        plt.show()
    else:
        print('Unknown vizualization.')


def normalize(df, stats, exclude=()):
    """
    Normalize raw_data.

    :param df: dataframe
    :param stats: tuple of mean and std
    :param exclude: column to exclude from normalization
    :return: processed dataframe
    """
    newdf = df.copy()

    for col in df.columns:
        if col not in exclude:
            series = df[col]
            mean, std = stats[col]
            series = (series - mean) / std
            newdf[col] = series

    return newdf


def get_stats(df, path='./'):
    """
    Compute mean and standard deviation for each column in the dataframe.

    :param df: dataframe
    :return: dictionary
    """
    stats = {}

    for col in df.columns:
        if col != "datetime":
            series = df[col]
            mean = series.mean()
            std = series.std()
            stats[col] = (mean, std)

    filename = os.path.join(path, 'stats.json')
    save_json(data=stats, filename=filename)

    return stats


def filter(df, column, threshold):
    """
    Filter dataframe based on a single column and its threshold if the column exists.

    :param df: dataframe
    :param column: column name to filter
    :param threshold: threshold value for filtering
    :return: filtered dataframe if column exists, otherwise original dataframe
    """
    if column in df.columns:
        if threshold is not None:
            df = df[df[column] > threshold]
        else:
            df.drop(column, axis="columns", inplace=True)

    return df


def aggregate(df, grp="1min", func=lambda x: x):
    """
    Resample dataframe based on the provided frequency and aggregate using the specified function.

    :param df: dataframe
    :param grp: resampling frequency ('1min' -> original)
    :param func: aggregation function (lambda x: x -> no aggregation)
    :return: aggregated dataframe
    """
    df = df.set_index("DATETIME")

    if grp:
        df = df.resample(grp)
        df = df.apply(func)
        df = df.dropna()

    df = df.sort_index()

    return df


def get_optim(name, model, lr):
    """
    Get optimizer object based on name, model, and learning rate.

    :param name: str
    :param model: model
    :param lr: float
    :return: optimizer object
    """
    optim_class = getattr(optim, name)
    optimizer = optim_class(model.parameters(), lr=lr)

    return optimizer


def get_sched(name, step_size, gamma, optimizer):
    """
    Get scheduler object based on name, step size, gamma, and optimizer.

    :param name: str
    :param step_size: int
    :param gamma: gamma float
    :param optimizer: optimizer object
    :return: scheduler object
    """
    sched_class = getattr(sched, name)
    scheduler = sched_class(optimizer, step_size, gamma)

    return scheduler