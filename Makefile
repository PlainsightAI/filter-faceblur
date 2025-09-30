# ---------------------------------
# Repo-specific variables
# ---------------------------------

IMAGE ?= us-west1-docker.pkg.dev/plainsightai-prod/oci/filter-faceblur

# Define these for consistency in the repo
REPO_NAME ?= filter-faceblur
REPO_NAME_SNAKECASE ?= filter_faceblur
REPO_NAME_PASCALCASE ?= FilterFaceblur

# Unique pipeline configuration for this repo
PIPELINE := \
	- VideoIn \
		--sources file://data/video-01.mp4!loop \
	- $(REPO_NAME_SNAKECASE).filter.$(REPO_NAME_PASCALCASE) \
	- Webvis

# ---------------------------------
# Repo-specific targets
# ---------------------------------

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install:  ## Install package with dev dependencies
	pip install -e .[dev] \
		--index-url https://python.openfilter.io/simple \
		--extra-index-url https://pypi.org/simple

.PHONY: run
run:  ## Run locally with supporting Filters in other processes
	openfilter run ${PIPELINE}

.PHONY: test
test:  ## Run unit tests
	pytest -vv -s tests/ --junitxml=results/pytest-results.xml

.PHONY: test-coverage
test-coverage:  ## Run unit tests and generate coverage report
	@mkdir -p Reports
	@pytest -vv --cov=tests --junitxml=Reports/coverage.xml --cov-report=json:Reports/coverage.json -s tests/
	@jq -r '["File Name", "Statements", "Missing", "Coverage%"], (.files | to_entries[] | [.key, .value.summary.num_statements, .value.summary.missing_lines, .value.summary.percent_covered_display]) | @csv'  Reports/coverage.json >  Reports/coverage_report.csv
	@jq -r '["TOTAL", (.totals.num_statements // 0), (.totals.missing_lines // 0), (.totals.percent_covered_display // "0")] | @csv'  Reports/coverage.json >>  Reports/coverage_report.csv

.PHONY: build-wheel
build-wheel:  ## Build python wheel
	python -m pip install setuptools build wheel twine setuptools-scm --index-url https://pypi.org/simple
	python -m build --wheel

.PHONY: clean
clean:  ## Delete all generated files and directories
	sudo rm -rf build/ cache/ dist/ $(REPO_NAME_SNAKECASE).egg-info/ telemetry/
	find . -name __pycache__ -type d -exec rm -rf {} +