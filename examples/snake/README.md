
This is the example for the snake game.

### Installation

```bash
pip install "openrl[selfplay]"
```

### Usage

```bash
python train_selfplay.py
```

### Evaluate JiDi submissions locally

```bash
python jidi_eval.py
```

## Submit to JiDi

Submition site: http://www.jidiai.cn/env_detail?envid=1.

Snake senarios: [here](https://github.com/jidiai/ai_lib/blob/7a6986f0cb543994277103dbf605e9575d59edd6/env/config.json#L94)
Original Snake environment: [here](https://github.com/jidiai/ai_lib/blob/master/env/snakes.py)




### Evaluate Google Research Football submissions for JiDi locally

If you want to evaluate your Google Research Football submissions for JiDi locally, please try to use tizero as illustrated [here](foothttps://github.com/OpenRL-Lab/TiZero#evaluate-jidi-submissions-locally).