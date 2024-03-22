# Crafter

## Installation

```bash
git clone git@github.com:danijar/crafter.git
git fetch origin pull/25/head:latest_gym
git checkout latest_gym
pip install -e .
```

## train

```bash
python train_crafter.py --config crafter_ppo.yaml
```
## render video

```bash
python render_crafter.py --config crafter_ppo.yaml
```

## render trajectory

* go to `openrl/envs/crafter/crafter.py`
* set `save_stats=True`

```python
self.env = crafter.Recorder(
    self.env, "crafter_traj",
    save_stats=True, # set this to be True
    save_episode=False,
    save_video=False,
)
```

* run the following command

```bash
python render_crafter.py --config crafter_ppo.yaml
```

* you can get the trajectory in `crafter_traj/stats.json1`. Following is an example of the stats file.
    
    ```json
    {"length": 143, "reward": 1.1, "achievement_collect_coal": 0, "achievement_collect_diamond": 0, "achievement_collect_drink": 15, "achievement_collect_iron": 0, "achievement_collect_sapling": 0, "achievement_collect_stone": 0, "achievement_collect_wood": 0, "achievement_defeat_skeleton": 0, "achievement_defeat_zombie": 0, "achievement_eat_cow": 0, "achievement_eat_plant": 0, "achievement_make_iron_pickaxe": 0, "achievement_make_iron_sword": 0, "achievement_make_stone_pickaxe": 0, "achievement_make_stone_sword": 0, "achievement_make_wood_pickaxe": 0, "achievement_make_wood_sword": 0, "achievement_place_furnace": 0, "achievement_place_plant": 0, "achievement_place_stone": 0, "achievement_place_table": 0, "achievement_wake_up": 3}
    ```