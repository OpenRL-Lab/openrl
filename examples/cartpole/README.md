## How to Use

Users can train CartPole via:

```shell
python train_ppo.py
```


To train with [Dual-clip PPO](https://arxiv.org/abs/1912.09729):

```shell
python train_ppo.py --config dual_clip_ppo.yaml
```

If you want to save checkpoints, try to train with Callbacks:

```shell
python train_ppo.py --config callbacks.yaml
```