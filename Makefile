DIRECTORIES = data results data/raw
RAWDATA = data/raw/diabetes.csv


data/train.parquet data/tune.parquet data/test.parquet &: data/base_data.parquet src/split_data.py | ${DIRECTORIES}
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
