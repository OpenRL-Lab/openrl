__TITLE__ = "openrl"
__VERSION__ = "v0.0.15"
__DESCRIPTION__ = "Distributed Deep RL Framework"
__AUTHOR__ = "OpenRL Contributors"
__EMAIL__ = "huangshiyu@4paradigm.com"
__version__ = __VERSION__

import platform

python_version_list = list(map(int, platform.python_version_tuple()))
assert python_version_list >= [
    3,
    8,
    0,
], (
    "OpenRL requires Python 3.8 or newer, but your Python is"
    f" {platform.python_version()}"
)
