
RAWDATA = data/raw/diabetes.csv

TRAIN_TEST_TUNE = data/train.parquet data/tune.parquet data/test.parquet
DIRECTORIES = data results data/raw data/xgboost_model results/xgboost_model

MODEL_GRAPHS = results/xgboost_model/pred_vs_actual.png results/xgboost_model/feature_importance.png results/xgboost_model/feature_importance_bar.png

${MODEL_GRAPHS} &: data/xgboost_model/model.ubj data/train.parquet src/examine_xgb_model.py | ${DIRECTORIES}
	python -m src.examine_xgb_model

data/xgboost_model/model.ubj: data/train.parquet data/tune.parquet src/build_gbm_model.py src/features.py src/xgb_prepare_data.py src/compare_predictions.py | ${DIRECTORIES}
	python -m src.build_gbm_model

# Create train, tune, and test parquet files
${TRAIN_TEST_TUNE} &: data/base_data.parquet src/split_data.py | ${DIRECTORIES}
	python -m src.split_data

# Generate base data parquet files
data/base_data.parquet: ${RAWDATA} ./src/read_raw_data.py | ${DIRECTORIES}
	python -m src.read_raw_data ${RAWDATA}

# Create the directories
${DIRECTORIES}:
	mkdir -p $@

# Build the docker image
docker-build:
	docker build -t base-python-ds .

clean:
	find ./data -type f ! -wholename "./data/raw/*" -exec rm -f {} +
	find ./results -type f -exec rm -f {} +
