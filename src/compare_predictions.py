from dataclasses import dataclass

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class RegressionMetrics:
    mse: float
    pearson_corr: float
    spearman_corr: float
    r2: float


def evaluate_regression(predictions, targets, weights=None):
    mse = mean_squared_error(targets, predictions, sample_weight=weights)
    pearson_corr, _ = pearsonr(targets, predictions)
    spearman_corr, _ = spearmanr(targets, predictions)
    r2 = r2_score(targets, predictions, sample_weight=weights)

    return RegressionMetrics(mse, pearson_corr, spearman_corr, r2)
