__TITLE__ = "openrl"
__VERSION__ = "v0.0.8"
__DESCRIPTION__ = "Distributed Deep RL Framework"
__AUTHOR__ = "OpenRL Contributors"
__EMAIL__ = "huangshiyu@4paradigm.com"
__version__ = __VERSION__

import platform

python_version = platform.python_version()
assert (
    python_version >= "3.8"
), f"OpenRL requires Python 3.8 or newer, but your Python is {python_version}"
