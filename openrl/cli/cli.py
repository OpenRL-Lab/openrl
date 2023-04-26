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

import click
from click.core import Context, Option
from termcolor import colored

from openrl import __AUTHOR__, __EMAIL__, __TITLE__, __VERSION__
from openrl.utils.util import get_system_info


def red(text: str):
    return colored(text, "red")


def print_version(
    ctx: Context,
    param: Option,
    value: bool,
) -> None:
    if not value or ctx.resilient_parsing:
        return
    click.secho(f"{__TITLE__.upper()} version: {red(__VERSION__)}")
    click.secho(f"Developed by {__AUTHOR__}, Email: {red(__EMAIL__)}")
    ctx.exit()


def print_system_info(
    ctx: Context,
    param: Option,
    value: bool,
) -> None:
    if not value or ctx.resilient_parsing:
        return
    info_dict = get_system_info()
    for key, value in info_dict.items():
        click.secho(f"- {key}: {red(value)}")
    ctx.exit()


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--version",
    is_flag=True,
    callback=print_version,
    expose_value=False,
    is_eager=True,
    help="Show package's version information.",
)
@click.option(
    "--system_info",
    is_flag=True,
    callback=print_system_info,
    expose_value=False,
    is_eager=True,
    help="Show system information.",
)
@click.option(
    "--mode",
    prompt="Choose execution mode",
    type=click.Choice(
        [
            "train",
        ]
    ),
    help="execution mode",
)
@click.option(
    "--env",
    prompt="Please enter environment name",
    type=str,
    help="RL environment name",
)
@click.option(
    "--env_step",
    type=int,
    default=20000,
    help="Maximum collected environment steps for training",
)
def run(mode: str, env: str, env_step: int):
    if mode == "train":
        from openrl.cli.train import train_agent

        train_agent(env, total_time_steps=env_step)
    else:
        raise NotImplementedError
