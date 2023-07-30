#!/bin/bash

if [[ $1 = "test" ]]; then
  twine upload dist/* -r testpypi
else
  twine upload dist/*
fi