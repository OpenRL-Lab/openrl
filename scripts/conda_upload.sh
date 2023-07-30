#!/bin/bash

export PATH="~/anaconda3/bin:$PATH"

VERSION=$(python setup.py --version)
echo $VERSION
deactivate
conda init zsh
conda activate base
anaconda upload --user openrl ~/anaconda3/conda-bld/osx-64/openrl-v${VERSION}-py38_0.tar.bz2
