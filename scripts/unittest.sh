#!/bin/bash

pytest tests --cov=openrl --cov-report=xml -m unittest --cov-report=term-missing --durations=0 -v --color=yes
