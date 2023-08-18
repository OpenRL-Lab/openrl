## Installation

Installation guide for Linux:

- `pip install pysc2`
- Download `StarCraftII.zip` from [Google Drive](https://drive.google.com/drive/folders/1umnlFotrXdEnmTUqfzGoJfJ7kAKf-eKO).
- unzip `StarCraftII.zip` to `~/StarCraftII/`: `unzip StarCraftII.zip -d ~/`
- If something is wrong with protobuf, you can do this: `pip install protobuf==3.20.3`

## Usage

Train SMAC with [MAPPO](https://arxiv.org/abs/2103.01955) algorithm:

`python train_ppo.py --config smac_ppo.yaml`

## Render replay on Mac

