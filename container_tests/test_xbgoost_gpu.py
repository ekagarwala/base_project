import xgboost as xgb
import numpy as np

# Create a simple dataset
X = np.random.rand(100, 10)
y = np.random.randint(2, size=100)

# Create a DMatrix
dtrain = xgb.DMatrix(X, label=y)

# Set parameters for GPU training
params = {
    'device': 'cuda',  # Use GPU acceleration
    'tree_method': 'hist',  # Use GPU for training
}

try:
    # Train a simple model
    bst = xgb.train(params, dtrain, num_boost_round=10)
    print("GPU is available and XGBoost is using it.")
except xgb.core.XGBoostError as e:
    print("GPU is not available or XGBoost is not configured to use it.")
    print("Error:", e)