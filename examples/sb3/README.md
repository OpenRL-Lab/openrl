Load and use [stable-baseline3 models](https://huggingface.co/sb3) from huggingface.

## Installation

```bash
pip install huggingface-tool
pip install rl_zoo3
```

## Download sb3 model from huggingface

```bash
htool save-repo sb3/ppo-CartPole-v1 ppo-CartPole-v1
```

## Use OpenRL to load the model trained by sb3 and then evaluate it

```bash
python test_model.py
```

## Use OpenRL to load the model trained by sb3 and then train it

```bash
python train_ppo.py
```


