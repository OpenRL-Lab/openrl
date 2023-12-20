## Installation

 
Install envpool with:

``` shell
pip install envpool
```

Note 1: envpool only supports Linux operating system.

## Usage

You can use `OpenRL` to train Cartpole (envpool) via:

``` shell
PYTHON_PATH train_ppo.py
```

You can also add custom wrappers in `envpool_wrapper.py`. Currently we have `VecAdapter` and `VecMonitor` wrappers.