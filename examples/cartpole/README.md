## How to Use

Users can train CartPole via:

```shell
python train_ppo.py --config ppo.yaml
```


To train with [Dual-clip PPO](https://arxiv.org/abs/1912.09729):

```shell
python train_ppo.py --config dual_clip_ppo.yaml
```

To train with [A2C](https://arxiv.org/abs/1602.01783) algorithm:

```shell
python train_a2c.py
```

If you want to evaluate the agent during training and save the best model and save checkpoints, try to train with callbacks:

```shell
python train_ppo.py --config callbacks.yaml
```

More details about callbacks can be found in [Callbacks](https://openrl-docs.readthedocs.io/en/latest/callbacks/index.html).