DIRECTORIES = data results data/raw

${DIRECTORIES}:
	mkdir -p $@

# Build the docker image
docker-build:
	docker build -t base-python-ds .
