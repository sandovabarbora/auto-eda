SHELL := /bin/bash
# Choose python3 on Unix, python on Windows
ifeq ($(OS),Windows_NT)
 PYTHON := python
 VENV_DIR := venv\\Scripts
else
 PYTHON := python3
 VENV_DIR := venv/bin
endif

# Phony targets
.PHONY: help setup install activate clean run test lint

help:
	@echo "Available commands:"
	@echo " make setup - Create virtual environment and upgrade pip"
	@echo " make install - Install dependencies (uses venv directly, no need to activate)"
	@echo " make activate - Print instructions for manually activating the virtual environment"
	@echo " make run - Run the AutoEDA Streamlit application"
	@echo " make clean - Remove the virtual environment and cache files"
	@echo " make deepclean - Comprehensive removal of all build artifacts and caches"

setup:
	@echo "Setting up environment..."
	@if [ ! -d venv ]; then \
		echo "No virtual environment found. Creating 'venv'..."; \
		$(PYTHON) -m venv venv; \
	fi
	@$(VENV_DIR)/pip install --upgrade pip
	@echo "Setup complete!"
	@echo "=============================================================================="
	@echo "To activate your virtual environment manually, run one of the following:"
ifeq ($(OS),Windows_NT)
	@echo " call venv\\Scripts\\activate.bat (Windows CMD)"
	@echo " or .\\venv\\Scripts\\Activate.ps1 (Windows PowerShell)"
else
	@echo " source venv/bin/activate (Unix/Linux/macOS)"
endif
	@echo "Then run 'make install' to install dependencies."
	@echo "To deactivate the environment, run 'deactivate'."
	@echo "To run the application without activating, use 'make run'."
	@echo "=============================================================================="

install:
	@echo "Installing dependencies..."
	@$(VENV_DIR)/pip install -r requirements.txt
	@echo "Dependencies installed!"

activate:
	@echo "To activate your virtual environment, run one of the following:"
ifeq ($(OS),Windows_NT)
	@echo " call venv\\Scripts\\activate.bat (Windows CMD)"
	@echo " or .\\venv\\Scripts\\Activate.ps1 (Windows PowerShell)"
else
	@echo " source venv/bin/activate (Unix/Linux/macOS)"
endif
	@echo "To deactivate when finished, run 'deactivate'."

clean:
	@echo "Cleaning up basic project artifacts..."
	@rm -rf venv 2>/dev/null || true
	@rm -rf __pycache__ 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Virtual environment and cache files removed."

deepclean: clean
	@echo "Performing deep clean..."
	@rm -rf build 2>/dev/null || true
	@rm -rf dist 2>/dev/null || true
	@rm -rf *.egg-info 2>/dev/null || true
	@rm -rf .pytest_cache 2>/dev/null || true
	@rm -rf .coverage 2>/dev/null || true
	@rm -rf htmlcov 2>/dev/null || true
	@rm -rf .mypy_cache 2>/dev/null || true
	@rm -rf *.html 2>/dev/null || true # Remove generated HTML reports
	@echo "Deep clean completed. All build artifacts and caches removed."

run:
	@echo "Running AutoEDA Streamlit application..."
ifeq ($(OS),Windows_NT)
	@$(VENV_DIR)\\streamlit run app.py
else
	@$(VENV_DIR)/streamlit run app.py
endif

generate-sample-report:
	@echo "Generating a sample EDA report from test data..."
ifeq ($(OS),Windows_NT)
	@$(VENV_DIR)\\python -c "from autoeda import run_autoeda; run_autoeda('path/to/sample_data.csv', output_file='sample_report.html')"
else
	@$(VENV_DIR)/python -c "from autoeda import run_autoeda; run_autoeda('path/to/sample_data.csv', output_file='sample_report.html')"
endif
	@echo "Sample report generated as 'sample_report.html'"