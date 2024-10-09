import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import matplotlib.pyplot as plt
import pandas as pd
import os
import json


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


class MaskedLogCosh(nn.Module):
    """
    MaskedMSELoss that utilizes masks
    """
    def __init__(self):
        super(MaskedLogCosh, self).__init__()

    def forward(self, pred, true, mask):
        pred, mask = pred.squeeze(), mask.squeeze()
        # Compute element-wise squared difference
        loss = torch.log(torch.cosh(pred - true)) * mask
        # Compute the mean loss only over non-masked elements
        loss = loss.sum() / (mask.sum() + 1e-8)

        return loss


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

    elif vizualization == 'training_predictions' or vizualization == 'testing_predictions':
        y_true, y_pred = args
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, color='blue', edgecolor='k', alpha=0.7)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', lw=2)
        plt.title('Predicted vs Actual')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.show()
    else:
        print('Unknown vizualization.')


def normalize(df, stats, exclude=None):
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
