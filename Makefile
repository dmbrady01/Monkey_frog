# HELP
# This will output the help for each task
# thanks to https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
.PHONY: help

help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help

create: ## Creates the containers needed for monkey_frog (but does not start them)
	docker-compose -f docker-compose.yaml up --no-start

up: ## Compose up
	docker-compose -f docker-compose.yaml up -d

up-no-d: ## Compose up but no detach
	docker-compose -f docker-compose.yaml up

down: ## Compose down
	docker-compose down -v --remove-orphans

bash: ## Compose run but run bash
	docker-compose -f docker-compose.yaml run --entrypoint /bin/bash app

ipython: ## Compose run but run ipython
	docker-compose -f docker-compose.yaml run --entrypoint ipython app

run: ## Run process_data.py. Set file=myfile.json to set the parameter file
	docker-compose -f docker-compose.yaml run app -f $(file)
