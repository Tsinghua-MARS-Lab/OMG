PYTHON ?= python3
UV ?= uv
PYTHONPATH := src
TORCH_INDEX_URL ?= https://download.pytorch.org/whl/cu124
PYPI_INDEX_URL ?=
PIP_INDEX_ARGS := $(if $(PYPI_INDEX_URL),--index-url $(PYPI_INDEX_URL),)
EXP ?= 100m
DATA ?= omg_data_lerobot
TRAINER ?= 1gpu

.PHONY: venv install install-cn install-dev compile compose smoke test

venv:
	$(UV) venv --python 3.10 .venv

install:
	$(UV) pip install torch==2.6.0 --index-url $(TORCH_INDEX_URL)
	$(UV) pip install -e ".[all]" $(PIP_INDEX_ARGS)

install-cn:
	$(MAKE) install PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

install-dev: install
	$(UV) pip install -e ".[dev]" $(PIP_INDEX_ARGS)

compile:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall src tests

compose:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m omg.cli.generation.train exp=$(EXP) data=$(DATA) logger=none --cfg job

smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m omg.cli.generation.train exp=$(EXP) data=$(DATA) logger=none trainer=$(TRAINER) data.limit_each_trainset=2 trainer.max_steps=2 trainer.limit_val_batches=0 trainer.num_sanity_val_steps=0

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q
