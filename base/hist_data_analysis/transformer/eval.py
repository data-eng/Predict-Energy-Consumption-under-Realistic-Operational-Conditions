import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from loader import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(dirs):

    cfn = utils.get_path(dirs=dirs, name="test_checkpoints.json")

    test_preds = utils.load_json(filename=cfn)

    y_true = test_preds['true_values']
    y_pred = test_preds['pred_values']

    nan_indices = [i for i, value in enumerate(y_true) if value == -1]
    y_true = [value for idx, value in enumerate(y_true) if idx not in nan_indices]
    y_pred = [value for idx, value in enumerate(y_pred) if idx not in nan_indices]

    # Convert lists to numpy arrays for convenience
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 1. Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"Mean Absolute Error (MAE)      : {mae:.5f}")
    print(f"Mean Squared Error (MSE)       : {mse:.5f}")
    print(f"Root Mean Squared Error (RMSE) : {rmse:.5f}")
    print(f"R-squared (R²)                 : {r2:.5f}")

    # 2. Predicted vs Actual Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='blue', edgecolor='k', alpha=0.7)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', lw=2)
    plt.title('Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.show()

    # 3. Residuals Plot
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, color='blue', edgecolor='k', alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--', lw=2)
    plt.title('Residuals Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.show()

    # 4. Distribution of Residuals
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='blue', bins=10)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


evaluate(dirs=["models", "13"])
