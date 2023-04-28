SHELL=/bin/bash
PROJECT_NAME=openrl
PROJECT_PATH=${PROJECT_NAME}/
PYTHON_FILES = $(shell find setup.py ${PROJECT_NAME} tests examples -type f -name "*.py")

check_install = python3 -c "import $(1)" || pip3 install $(1) --upgrade
check_install_extra = python3 -c "import $(1)" || pip3 install $(2) --upgrade

test:
	./scripts/unittest.sh

lint:
	$(call check_install, ruff)
	ruff ${PYTHON_FILES} --select=E9,F63,F7,F82 --show-source
	ruff ${PYTHON_FILES} --exit-zero

format:
	$(call check_install, isort)
	$(call check_install, black)
	# Sort imports
	isort ${PYTHON_FILES}
	# Reformat using black
	black ${PYTHON_FILES} --preview

commit-checks: format lint