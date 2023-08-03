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


import re
import tempfile

import yaml
from jinja2 import Environment, Template, meta
from jsonargparse import ActionConfigFile, ArgumentParser


class ProcessYamlAction(ActionConfigFile):
    def __call__(self, parser, cfg, values, option_string=None):
        # Read the original YAML file
        assert isinstance(values, str) and values.endswith(".yaml")
        with open(values, "r") as file:
            content = file.read()

        # Initialize global variables
        global_variables = {}

        # Extract globals section using regular expressions if present
        globals_match = re.search(
            r"^globals:\n((?:  [^\n]*\n)*)", content, re.MULTILINE
        )
        if globals_match:
            global_variables_yaml = globals_match.group(1)
            global_variables = yaml.safe_load("globals:\n" + global_variables_yaml).get(
                "globals", {}
            )

        # Create a Jinja2 environment
        env = Environment()

        # Parse original content without rendering to find all variable names
        parsed_content = env.parse(content)
        all_variables = meta.find_undeclared_variables(parsed_content)

        # Check that all variables are defined in the global variables
        undefined_variables = all_variables - set(global_variables.keys())
        if undefined_variables:
            # Iterate through the undefined variables and find their line numbers
            error_messages = []
            for variable in undefined_variables:
                line_number = next(
                    (
                        i + 1
                        for i, line in enumerate(content.splitlines())
                        if "{{ " + variable + " }}" in line
                    ),
                    "Unknown",
                )
                error_messages.append(
                    f"Undefined global variable: '{variable}' at line {line_number}"
                )
            raise ValueError("\n".join(error_messages))

        # Remove 'globals' section and its variables from the YAML content
        content_without_globals = re.sub(
            r"^globals:\n((?:  [^\n]*\n)*)", "", content, flags=re.MULTILINE
        )

        # Render content without globals using Jinja2 with global variables
        template_without_globals = env.from_string(content_without_globals)
        rendered_content = template_without_globals.render(global_variables)

        # Load the rendered content as a dictionary
        data = yaml.safe_load(rendered_content)

        # Write the result to a temporary file
        with tempfile.NamedTemporaryFile("w", delete=True, suffix=".yaml") as temp_file:
            yaml.dump(data, temp_file)
            temp_file.seek(0)  # Move to the beginning of the file
            # Use the default behavior of ActionConfigFile to handle the temporary file
            super().__call__(parser, cfg, temp_file.name, option_string)
