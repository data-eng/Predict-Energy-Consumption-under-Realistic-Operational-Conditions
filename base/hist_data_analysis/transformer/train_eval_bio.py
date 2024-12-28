import random
from torch.utils.data import DataLoader
from model import Transformer
from sklearn.metrics import r2_score
from loader import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Device is {device}')


def test(test_data, criterion, model, seed, dirs, y_label, stats, visualize=False):
    mfn = utils.get_path(dirs=dirs, name="transformer.pth")
    model.load_state_dict(torch.load(mfn))

    model.to(device)
    torch.manual_seed(seed)

    true_values, pred_values = [], []

    results = {'seed': seed}

    logger.info(f"\nTraining with seed {seed} just started...")

    model.eval()
    total_test_loss = 0.0

    with torch.no_grad():
        for X, y, mask_X, mask_y in test_data:
            X, y, mask_X, mask_y = X.to(device), y.to(device), mask_X.to(device), mask_y.to(device)
            y, mask_y = y.squeeze(), mask_y.squeeze()

            if len(X.shape) == 2:
                X = X.unsqueeze(0)
                mask_X = mask_X.unsqueeze(0)

            y_pred = model(X, mask_X)

            true_values.append(y.cpu().numpy())
            pred_values.append(y_pred.detach().cpu().numpy())

            test_loss = criterion(pred=y_pred, true=y, mask=mask_y)
            total_test_loss += test_loss.item()

    avg_test_loss = total_test_loss / len(test_data)

    true_values_list = [val.item() for val in true_values]
    pred_values_list = [val.item() for val in pred_values]

    logger.info(f"Testing Loss: {avg_test_loss:.6f}")

    results['test_loss'] = avg_test_loss
    results['true_values'] = true_values_list
    results['pred_values'] = pred_values_list

    cfn = utils.get_path(dirs=dirs, name="test_checkpoints.json")

    utils.save_json(data=results, filename=cfn)

    # Get the indices where values are NaN
    # print(len(true_values_list))
    nan_indices = [i for i, value in enumerate(true_values_list) if value == -1]
    true_values_list = [value for idx, value in enumerate(true_values_list) if idx not in nan_indices]
    true_values_list = utils.unnormalize(y=true_values_list, stats=stats, column=y_label)
    pred_values_list = [value for idx, value in enumerate(pred_values_list) if idx not in nan_indices]
    pred_values_list = utils.unnormalize(y=pred_values_list, stats=stats, column=y_label)
    # print(len(true_values_list))

    if visualize:
        utils.visualize('testing_predictions', true_values_list, pred_values_list)

    logger.info(f'\nTesting with seed {seed} complete!\n')

    return avg_test_loss


def train(data, epochs, patience, lr, criterion, model, optimizer, scheduler, seed, dirs, y_label, stats, visualize=False):
    model.to(device)
    train_data, val_data = data
    batches = len(train_data)
    optimizer = utils.get_optim(optimizer, model, lr)
    scheduler = utils.get_sched(*scheduler, optimizer)
    torch.manual_seed(seed)

    best_val_loss = float('inf')
    stationary = 0
    train_losses, val_losses = [], []

    checkpoints = {'seed': seed,
                   'epochs': 0,
                   'best_epoch': 0,
                   'best_train_loss': float('inf'),
                   'best_val_loss': float('inf'),
                   'true_vals': [],
                   'pred_vals': []}

    logger.info(f"\nTraining with seed {seed} just started...")

    # One forward pass of all training data
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        true_values, pred_values = [], []

        for _, (X, y, mask_X, mask_y) in enumerate(train_data):
            X, y, mask_X, mask_y = X.to(device), y.to(device), mask_X.to(device), mask_y.to(device)
            y, mask_y = y.squeeze(), mask_y.squeeze()
            y_pred = model(X, mask_X)

            train_loss = criterion(pred=y_pred, true=y, mask=mask_y)

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

        # One validation pass to activate early stopping if needed
        with torch.no_grad():
            for X, y, mask_X, mask_y in val_data:
                X, y, mask_X, mask_y = X.to(device), y.to(device), mask_X.to(device), mask_y.to(device)
                y, mask_y = y.squeeze(), mask_y.squeeze()
                y_pred = model(X, mask_X)

                val_loss = criterion(pred=y_pred, true=y, mask=mask_y)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / batches
        val_losses.append(avg_val_loss)

        true_values = np.concatenate(true_values)
        pred_values = np.concatenate(pred_values)

        true_values_list = true_values.tolist()
        pred_values_list = pred_values.tolist()

        logger.info(
            f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stationary = 0

            logger.info(f"~ New best val found!")

            mfn = utils.get_path(dirs=dirs, name="transformer.pth")
            torch.save(model.state_dict(), mfn)

            checkpoints.update({'best_epoch': epoch + 1,
                                'best_train_loss': avg_train_loss,
                                'best_val_loss': best_val_loss,
                                'true_vals': true_values_list,
                                'pred_vals': pred_values_list})
        else:
            stationary += 1

        if stationary >= patience:
            logger.info(f"Early stopping after {epoch + 1} epochs without improvement. Patience is {patience}.")
            break

        scheduler.step()

    # Store predictions and losses and print diagrams for the training data
    cfn = utils.get_path(dirs=dirs, name="train_checkpoints.json")
    checkpoints.update({'epochs': epoch + 1})
    utils.save_json(data=checkpoints, filename=cfn)

    # Get the indices where values are NaN
    # print(len(true_values_list))
    nan_indices = [i for i, value in enumerate(true_values_list) if value == -1]
    true_values_list = [value for idx, value in enumerate(true_values_list) if idx not in nan_indices]
    true_values_list = utils.unnormalize(y=true_values_list, stats=stats, column=y_label)
    pred_values_list = [value for idx, value in enumerate(pred_values_list) if idx not in nan_indices]
    pred_values_list = utils.unnormalize(y=pred_values_list, stats=stats, column=y_label)
    # print(len(true_values_list))

    r2 = r2_score(true_values_list, pred_values_list)
    logger.info(f"R-squared (R²): {r2:.5f}")

    if visualize:
        cfn = utils.get_path(dirs=dirs, name="train_losses.json")
        utils.save_json(data=train_losses, filename=cfn)

        utils.visualize('losses', train_losses, val_losses)
        utils.visualize('training_predictions', true_values_list, pred_values_list)

    logger.info(
        f'\nTraining with seed {seed} complete!\nFinal Training Loss: {avg_train_loss:.6f} & Validation Loss: {best_val_loss:.6f}\n')

    return avg_train_loss, best_val_loss


def main_loop(time_repr, seed, dirs):
    seq_len = 10
    batch_size = 8

    data_dir = "../../../data_creation/bio_data"
    ship_1_train = pd.read_csv(f'{data_dir}/ship_1_train.csv')
    ship_2_train = pd.read_csv(f'{data_dir}/ship_2_train.csv')
    ship_3_train = pd.read_csv(f'{data_dir}/ship_3_train.csv')

    data_train = pd.concat([ship_1_train, ship_2_train, ship_3_train],
                           keys=['ship_1', 'ship_2', 'ship_3'], names=['Source'], axis=0)

    data_train['datetime'] = pd.to_datetime(data_train['datetime'])

    y = 'ME_FO_consumption'

    # Load data, split into train, validation and testing set
    df_train, params, stats = load_bio(df=data_train, normalize=True, time_repr=time_repr, y=y)

    ship_1_train = df_train.loc['ship_1'].reset_index(drop=True)
    ship_2_train = df_train.loc['ship_2'].reset_index(drop=True)
    ship_3_train = df_train.loc['ship_3'].reset_index(drop=True)

    ds_train = TSDatasetBio(df1=ship_1_train, df2=ship_2_train, df3=ship_3_train,
                            seq_len=seq_len, X=params["X"], t=params["t"], y=y)

    ds_train, ds_val = split(ds_train, vperc=0.2)

    dl_train = DataLoader(ds_train, batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size, shuffle=False)

    # Create the regressor
    model = Transformer(in_size=len(params["X"]) + len(params["t"]),
                        sequence_len=seq_len,
                        out_size=1,
                        nhead=2,
                        num_layers=2,
                        dim_feedforward=1024,
                        dropout=0)

    # Train model
    _, _ = train(data=(dl_train, dl_val),
                 epochs=3 , #500,
                 patience=10,
                 lr=5e-4,
                 # lr=5e-3,
                 # criterion=utils.MaskedLogCosh(),
                 criterion=utils.MaskedMSELoss(),
                 model=model,
                 optimizer="AdamW",
                 scheduler=("StepLR", 1.0, 0.98),
                 seed=seed,
                 dirs=dirs,
                 y_label=y,
                 stats=stats,
                 visualize=True)

    # Evaluate on testing sets

    ship_1_test = pd.read_csv(f'{data_dir}/ship_1_test.csv')
    ship_2_test = pd.read_csv(f'{data_dir}/ship_2_test.csv')
    ship_3_test = pd.read_csv(f'{data_dir}/ship_3_test.csv')

    data_test = pd.concat([ship_1_test, ship_2_test, ship_3_test],
                          keys=['ship_1', 'ship_2', 'ship_3'], names=['Source'], axis=0)
    data_test['datetime'] = pd.to_datetime(data_test['datetime'])

    df_test, params, _ = load_bio(df=data_test, normalize=True, time_repr=time_repr, y=y)

    ship_1_test = df_test.loc['ship_1'].reset_index(drop=True)
    ship_2_test = df_test.loc['ship_2'].reset_index(drop=True)
    ship_3_test = df_test.loc['ship_3'].reset_index(drop=True)

    ds_test = TSDatasetBio(df1=ship_1_test, df2=ship_2_test, df3=ship_3_test,
                           seq_len=seq_len, X=params["X"], t=params["t"], y=y)
    
    # RESCALE DATA
    _ = test(test_data=ds_test,
             criterion=utils.MaskedMSELoss(),
             model=model,
             seed=seed,
             dirs=dirs,
             y_label=y,
             stats=stats,
             visualize=True)


def set_seed(seed):
    # Set the seed for generating random numbers in Python
    random.seed(seed)
    # Set the seed for generating random numbers in NumPy
    np.random.seed(seed)
    # Set the seed for generating random numbers in PyTorch (CPU)
    torch.manual_seed(seed)
    # If you are using GPUs, set the seed for generating random numbers on all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure that all operations on GPU are deterministic (if possible)
    torch.backends.cudnn.deterministic = True
    # Disable the inbuilt cudnn auto-tuner that finds the best algorithm to use for your hardware
    torch.backends.cudnn.benchmark = False


seed_num = 13
set_seed(seed_num)

main_loop(time_repr=(["month", "hour", "second"], ["sine", "sine", "sine"], ["cosine", "cosine", "cosine"],
                     [[(12, None, 0), (12, None, 0), (12, None, 0)], [(24, None, 0), (24, None, 0), (24, None, 0)],
                      [(60, None, 0), (60, None, 0), (60, None, 0)]]),
          seed=seed_num,
          dirs=["models", str(seed_num)])
