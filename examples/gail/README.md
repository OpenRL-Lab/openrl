## Prepare Dataset

Run following command to generate dataset for GAIL: `python gen_data.py`, then you will get a file named `data.pkl` in current folder.

## Train

Run following command to train GAIL: `python train_gail.py --config cartpole_gail.yaml`

With GAIL, we can even train the agent without expert action!
Run following command to train GAIL without expert action: `python train_gail.py --config cartpole_gail_without_action.yaml`