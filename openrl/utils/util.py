import platform
import random
import re
from typing import Dict

import gymnasium as gym
import numpy as np
import torch

import openrl


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


def check_v2(input, use_half=False, tpdv=None):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    if tpdv:
        output = output.to(**tpdv)
    if use_half:
        output = output.half()
    return output


def _t2n(x):
    return x.detach().cpu().numpy()


def get_system_info() -> Dict[str, str]:
    """
    Retrieve system and python env info for the current system.

    :return: Dictionary summing up the version for each relevant package
        and a formatted string.
    """

    env_info = {
        # In OS, a regex is used to add a space between a "#" and a number to avoid
        # wrongly linking to another issue on GitHub.
        "OS": re.sub(r"#(\d)", r"# \1", f"{platform.platform()} {platform.version()}"),
        "Python": platform.python_version(),
        "OpenRL": openrl.__version__,
        "PyTorch": torch.__version__,
        "GPU Enabled": str(torch.cuda.is_available()),
        "Numpy": np.__version__,
        "Gymnasium": gym.__version__,
    }
    return env_info
