#!/bin/bash

rm -r ./api_docs
sphinx-apidoc -o ./api_docs openrl --force -H OpenRL -A OpenRL_Contributors
python scripts/modify_api_docs.py