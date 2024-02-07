# Local installation
.PHONY: init clean lock update install

install: ## Initalise the virtual env installing deps
	pipenv install --dev

clean: ## Remove all the unwanted clutter
	find . -type d -name __pycache__ | xargs rm -rf
	find . -type d -name '*.egg-info' | xargs rm -rf
	pipenv clean

lock: ## Lock dependencies
	pipenv lock

update: ## Update dependencies (whole tree)
	pipenv update --dev

sync: ## Install dependencies as per the lock file
	pipenv sync --dev && pipenv clean

# Linting and formatting
.PHONY: lint format

lint: ## Lint files with flake and mypy
	pipenv run ruff check . --fix

format: ## Run black and isort
	pipenv run ruff format .

# Testing
.PHONY: test