## How to Use

Users can train the dialog task via:

```shell
python train_ppo.py --config nlp_ppo.yaml
```

Users can train the dialog task with deepspeed:

```shell
deepspeed train_ppo.py --config nlp_ppo.yaml


```

After the training, users can chat with the agent via:

```shell
python chat.py
```


### Chat with other agents

- Chat with [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B): `python chat_6b.py`