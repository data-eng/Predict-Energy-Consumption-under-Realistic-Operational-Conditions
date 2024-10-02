import logging
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import utils
from model import Transformer
from loader import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logger.info(f'Device is {device}')


def train(data, epochs, patience, lr, criterion, model, optimizer, scheduler, seed, y, dirs, visualize=False):
    model.to(device)
    train_data, val_data = data
    batches = len(train_data)
    optimizer = utils.get_optim(optimizer, model, lr)
    scheduler = utils.get_sched(*scheduler, optimizer)
    torch.manual_seed(seed)

    best_val_loss = float('inf')
    ylabel = y[0]
    stationary = 0
    train_losses, val_losses = [], []

    checkpoints = {'seed': seed, 
                   'epochs': 0, 
                   'best_epoch': 0, 
                   'best_train_loss': float('inf'), 
                   'best_val_loss': float('inf'), 
                   'precision_micro': 0, 
                   'precision_macro': 0, 
                   'precision_weighted': 0,
                   'recall_micro': 0, 
                   'recall_macro': 0, 
                   'recall_weighted': 0, 
                   'fscore_micro': 0,
                   'fscore_macro': 0, 
                   'fscore_weighted': 0}
    
    # logger.info(f"\nTraining with seed {seed} just started...")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        true_values, pred_values = [], []

        for _, (X, y, mask_X, mask_y) in enumerate(train_data):
            X, y, mask_X, mask_y = X.to(device), y.long().to(device), mask_X.to(device), mask_y.to(device)
            y_pred = model(X, mask_X)

            batch_size, seq_len, _ = y_pred.size()
            y_pred = y_pred.reshape(batch_size * seq_len, 1)
            y = y.reshape(batch_size * seq_len)
            mask_y = mask_y.reshape(batch_size * seq_len)

            y_pred = utils.mask(tensor=y_pred, mask=mask_y, id=0)
            y = utils.mask(tensor=y, mask=mask_y, id=0)
            
            train_loss = criterion(pred=y_pred, true=y)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.item()

            true_values.append(y.cpu().numpy())
            pred_values.append(y_pred.detach().cpu().numpy())

        avg_train_loss = total_train_loss / batches
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for X, y, mask_X, mask_y in val_data:
                X, y, mask_X, mask_y = X.to(device), y.long().to(device), mask_X.to(device), mask_y.to(device)
                y_pred = model(X, mask_X)

                batch_size, seq_len, _ = y_pred.size()
                y_pred = y_pred.reshape(batch_size * seq_len, 1)
                y = y.reshape(batch_size * seq_len)
                mask_y = mask_y.reshape(batch_size * seq_len)

                y_pred = utils.mask(tensor=y_pred, mask=mask_y, id=0)
                y = utils.mask(tensor=y, mask=mask_y, id=0)

                val_loss = criterion(pred=y_pred, true=y)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / batches
        val_losses.append(avg_val_loss)

        true_values = np.concatenate(true_values)
        pred_values = np.concatenate(pred_values)

        true_values_list = true_values.tolist()
        pred_values_list = pred_values.tolist()
        
        # logger.info(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stationary = 0

            # logger.info(f"New best val found! ~ Epoch [{epoch + 1}/{epochs}], Val Loss {avg_val_loss}")

            mfn = utils.get_path(dirs=dirs, name="transformer.pth")
            torch.save(model.state_dict(), mfn)

            checkpoints.update({'best_epoch': epoch+1, 
                                'best_train_loss': avg_train_loss, 
                                'best_val_loss': best_val_loss, 
                                **utils.get_prfs(true=true_values_list, pred=pred_values_list)})

            if visualize:
                utils.visualize()
                
        else:
            stationary += 1

        if stationary >= patience:
            # logger.info(f"Early stopping after {epoch + 1} epochs without improvement. Patience is {patience}.")
            break

        scheduler.step()

    cfn = utils.get_path(dirs=dirs, name="train_checkpoints.json")
    checkpoints.update({'epochs': epoch+1})
    utils.save_json(data=checkpoints, filename=cfn)
    
    if visualize:

        cfn = utils.get_path(dirs=dirs, name="train_losses.json")
        utils.save_json(data=train_losses, filename=cfn)

        utils.visualize()

    # logger.info(f'\nTraining with seed {seed} complete!\nFinal Training Loss: {avg_train_loss:.6f} & Validation Loss: {best_val_loss:.6f}\n')

    return avg_train_loss, best_val_loss


def main_loop(time_repr, seed, dirs):
    path = "../../../data_creation/data/aggr_3min.csv"
    seq_len = 10
    batch_size = 8

    df, params = load(path=path, normalize=True, time_repr=time_repr)

    '''
    df_prep = prepare(df, phase="train", ignore=params["ignore"])

    ds = TSDataset(df=df_prep, seq_len=seq_len, X=params["X"], t=params["t"], y=y_col, per_day=True)
    ds_train, ds_val = split(ds, vperc=0.2)

    dl_train = DataLoader(ds_train, batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size, shuffle=False)

    model = Transformer(in_size=len(params["X"])+len(params["t"]),
                        out_size=1,
                        nhead=1,
                        num_layers=1,
                        dim_feedforward=2048,
                        dropout=0)

    _, _ = train(data=(dl_train, dl_val),
                 epochs=300,
                 patience=30,
                 lr=5e-4,
                 criterion=utils.MaskedMSELoss(),
                 model=model,
                 optimizer="AdamW",
                 scheduler=("StepLR", 1.0, 0.98),
                 seed=seed,
                 y=y_col,
                 dirs=dirs,
                 visualize=True)
    '''


main_loop(time_repr=(["month", "hour"], ["sine", "sine"], ["cosine", "cosine"],
                     [[(12, None, 0), (12, None, 0)], [(24, None, 0), (24, None, 0)]]),
          seed=13,
          dirs=["models", "1", "13"])