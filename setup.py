#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The OpenRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""""

import os

import setuptools
from setuptools import setup


def get_install_requires() -> list:
    return [
        "setuptools>=67.0",
        "gymnasium",
        "click",
        "termcolor",
        "gym",
        "torch",
        "treevalue",
        "rich",
        "wandb",
        "seaborn",
        "jsonargparse",
        "imageio",
        "opencv-python",
        "pygame",
        "mujoco",
        "tqdm",
    ]


def get_extra_requires() -> dict:
    req = {
        "test": [
            "pytest",
            "pytest-cov",
            "mypy",
            "isort",
            "black",
            "ruff",
            "gpustat",
            "gym-super-mario-bros",
        ],
        "dev": ["build", "twine"],
        "mpe": ["pyglet==1.5.27"],
        "nlp": [
            "transformers==4.18.0",
            "datasets",
            "nltk",
            "evaluate",
            "icetk",
        ],
        "retro": ["gym-retro"],
        "super_mario": ["gym-super-mario-bros"],
    }
    return req


def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("openrl", "__init__.py"), "r").read().split()
    return init[init.index("__VERSION__") + 2][1:-1]


setup(
    name="openrl",
    version=get_version(),
    description="unified reinforcement learning framework",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    author="openrl contributors",
    author_email="huangsy1314@163.com",
    url="https://github.com/OpenRL-Lab/openrl",
    packages=setuptools.find_packages(),
    entry_points={"console_scripts": ["openrl=openrl.cli.cli:run"]},
    project_urls={
        "Code": "https://github.com/OpenRL-Lab/openrl",
        "Documentation": "https://openrl-docs.readthedocs.io/zh/latest/",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    keywords=(
        "reinforcement-learning multi-agent "
        "reinforcement-learning-algorithms pytorch machine-learning "
        "baselines toolbox python data-science gym gymnasium"
    ),
    python_requires=">=3.8",
    install_requires=get_install_requires(),
    extras_require=get_extra_requires(),
)
