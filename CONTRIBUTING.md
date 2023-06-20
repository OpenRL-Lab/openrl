## How to Contribute to OpenRL

[中文介绍](./docs/CONTRIBUTING_zh.md)

The OpenRL community welcomes anyone to contribute to the development of OpenRL, whether you are a developer or a user. Your feedback and contributions are our driving force! You can join the contribution of OpenRL in the following ways:

- As an OpenRL user, discover bugs in OpenRL and submit an [issue](https://github.com/OpenRL-Lab/openrl/issues/new/choose).
- As an OpenRL user, discover errors in the documentation of OpenRL and submit an [issue](https://github.com/OpenRL-Lab/openrl/issues/new/choose).
- Write test code to improve the code coverage of OpenRL (you can check the code coverage situation of OpenRL from [here](https://app.codecov.io/gh/OpenRL-Lab/openrl)). You can choose interested code snippets for writing test codes.
- As an open-source developer, fix existing bugs for OpenRL.
- As an open-source developer, add new environments and examples for OpenRL.
- As an open-source developer, add new algorithms for OpenRL.

## Contributing to OpenRL

Welcome to contribute to the development of OpenRL. We appreciate your contribution!

- If you want to contribute new features, please create a new [issue](https://github.com/OpenRL-Lab/openrl/issues/new/choose) first 
to discuss the implementation details of this feature. If the feature is approved by everyone, you can start implementing the code.
- You can also check for unimplemented features and existing bugs in [Issues](https://github.com/OpenRL-Lab/openrl/issues), 
reply in the corresponding issue that you want to solve it, and then start implementing the code.

After completing your code implementation, you need to pull the latest `main` branch and merge it.
After resolving any merge conflicts,
you can submit your code for merging into OpenRL's main branch through [Pull Request](https://github.com/OpenRL-Lab/openrl/pulls).

Before submitting a Pull Request, you need to complete [Code Testing and Code Formatting](#code-testing-and-code-formatting).

Then, your Pull Request needs to pass automated testing on GitHub.

Finally, at least one maintainer's review and approval are required before being merged into the main branch.

## Code Testing and Code Formatting

Before submitting a Pull Request, make sure that your code passes unit tests and conforms with OpenRL's coding style.

Firstly, you should install the test-related packages: `pip install -e ".[test]"`

Then, ensure that unit tests pass by executing `make test`.

Lastly, format your code by running `make format`.

> Tip: OpenRL uses [black](https://github.com/psf/black) coding style. 
You can install black plugins in your editor as shown in the [official website](https://black.readthedocs.io/en/stable/integrations/editors.html)
to help automatically format codes.