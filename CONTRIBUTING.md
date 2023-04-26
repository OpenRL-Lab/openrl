## 贡献者手册

欢迎更多的人参与到OpenRL的开发中来，我们非常欢迎您的贡献！

- 如果您想要贡献新的功能，请先在请先创建一个新的[issue](https://github.com/OpenRL-Lab/openrl/issues/new/choose)，
 以便我们讨论这个功能的实现细节。如果该功能得到了大家的认可，您可以开始进行代码实现。
- 您也可以在 [Issues](https://github.com/OpenRL-Lab/openrl/issues) 中查看未被实现的功能和仍然存的在bug，
在对应的issue中进行回复，说明您想要解决该issue，然后开始进行代码实现。

在您完成了代码实现之后，您需要拉取最新的`main`分支并进行合并。
解决合并冲突后，
您可以通过提交 [Pull Request](https://github.com/OpenRL-Lab/openrl/pulls) 
的方式将您的代码合并到OpenRL的main分支中。

在提交Pull Request前，您需要完成 [代码测试和代码格式化](#代码测试和代码格式化)。

然后，您的Pull Request需要通过GitHub上的自动化测试。

最后，需要得到至少一个开发人员的review和批准，才能被合并到main分支中。

## 代码测试和代码格式化

在您提交Pull Request之前，您需要确保您的代码通过了单元测试，并且符合OpenRL的代码风格。

首先，您需要安装测试相关的包：`pip install -e ".[test]"`

然后，您需要确保单元测试通过，这可以通过执行`make unittest`来完成。

然后，您需要执行`make format`来格式化您的代码。

最后，您需要执行`make commit-checks`来检查您的代码是否符合OpenRL的代码风格。

> 小技巧: OpenRL使用 [black](https://github.com/psf/black) 代码风格。
您可以在您的编辑器中安装black的[插件](https://black.readthedocs.io/en/stable/integrations/editors.html)，
来帮助您自动格式化代码。



