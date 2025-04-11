.PHONY: setup run clean lint format test venv docs help

# Python interpreter to use
PYTHON = python3

# Virtual environment directory
VENV = venv

# Entry point for the application
APP = app.py

# Default target
.DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@echo "  make setup     - Set up development environment (create virtual environment and install dependencies)"
	@echo "  make run       - Run the EDA application"
	@echo "  make clean     - Remove virtual environment and cache files"
	@echo "  make lint      - Run linting checks"
	@echo "  make format    - Format code with Black and isort"
	@echo "  make test      - Run tests"
	@echo "  make venv      - Create virtual environment only"
	@echo "  make docs      - Generate documentation"

# Set up development environment
setup: venv
	@echo "Installing dependencies..."
	@$(VENV)/bin/pip install -r requirements.txt
	@$(VENV)/bin/pip install black isort pylint pytest sphinx
	@echo "Setup complete."

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created."
	@echo "To activate the virtual environment, run 'source $(VENV)/bin/activate'."
	@echo "To deactivate, run 'deactivate'."
	@echo "To run the application, use 'make run' after activating the virtual environment."
	@echo "To clean up, use 'make clean'."
	@echo "To run tests, use 'make test' after activating the virtual environment."
	@echo "To format code, use 'make format' after activating the virtual environment."
	@echo "To run linting checks, use 'make lint' after activating the virtual environment."
	@echo "To generate documentation, use 'make docs' after activating the virtual environment."
	@echo "To build documentation, use 'make docs-build' after activating the virtual environment."


# Run the application
run:
	@echo "Starting EDA application..."
	@if [ -d "$(VENV)" ]; then \
		$(VENV)/bin/streamlit run $(APP); \
	else \
		streamlit run $(APP); \
	fi

# Clean up
clean:
	@echo "Removing virtual environment and cache files..."
	@rm -rf $(VENV)
	@rm -rf __pycache__
	@rm -rf .pytest_cache
	@rm -rf components/__pycache__
	@rm -rf utils/__pycache__
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete."

# Run linting checks
lint:
	@echo "Running linting checks..."
	@if [ -d "$(VENV)" ]; then \
		$(VENV)/bin/pylint *.py components/*.py utils/*.py; \
	else \
		pylint *.py components/*.py utils/*.py; \
	fi

# Format code
format:
	@echo "Formatting code with Black and isort..."
	@if [ -d "$(VENV)" ]; then \
		$(VENV)/bin/black . && $(VENV)/bin/isort .; \
	else \
		black . && isort .; \
	fi

# Run tests
test:
	@echo "Running tests..."
	@if [ -d "$(VENV)" ]; then \
		$(VENV)/bin/pytest; \
	else \
		pytest; \
	fi

# Build documentation
docs:
	@echo "Generating documentation..."
	@if [ ! -d "docs" ]; then \
		mkdir -p docs; \
	fi
	@if [ -d "$(VENV)" ]; then \
		cd docs && $(VENV)/bin/sphinx-quickstart -q -p "EDA Tool" -a "Author" -v "0.1"; \
	else \
		cd docs && sphinx-quickstart -q -p "EDA Tool" -a "Author" -v "0.1"; \
	fi
	@echo "Documentation setup complete. Customize docs/conf.py and run 'make docs-build' to build."

# Build documentation after setup
docs-build:
	@echo "Building documentation..."
	@if [ -d "$(VENV)" ]; then \
		cd docs && $(VENV)/bin/sphinx-build -M html . _build; \
	else \
		cd docs && sphinx-build -M html . _build; \
	fi
	@echo "Documentation built in docs/_build/html/"