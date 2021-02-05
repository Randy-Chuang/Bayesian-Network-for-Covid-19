# -*- coding: utf-8 -*-
# @Filename : Makefile
# @Description: Makefile for term project.
# @Date : 2021-January
# @Project: Early detection of Covid-19 using BN (AI Term project)
# @AUTHOR : Randy Chuang

PYTHON = python3
PYDIR = py3

SHELL_PKG = python3-venv python3-tk
PIP_PRE_INSTALL = wheel
DEPENDENCIES = requirement.txt
PIP_INSTALL = -r $(DEPENDENCIES) --no-cache-dir

all: build training analysis

training: 
	. $(PYDIR)/bin/activate; \
	$(PYTHON) training.py

analysis: 
	. $(PYDIR)/bin/activate; \
	$(PYTHON) analysis.py

build: 
ifeq ( , $(wildcard $(PYDIR)))
	sudo apt-get install $(SHELL_PKG)
	$(PYTHON) -m venv $(PYDIR)
	. $(PYDIR)/bin/activate; \
	pip install $(PIP_PRE_INSTALL); \
	pip install $(PIP_INSTALL)
endif

clean: 
	rm -rf model/ dataset/ $(PYDIR)

