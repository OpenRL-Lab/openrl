## Prepare Dataset

Go to `examples/gail` folder.
Run following command to generate dataset for GAIL: `python gen_data.py`, then you will get a file named `data.pkl` in current folder.
Then copy the `data.pkl` to current folder.

## Train

Run following command to train behavior cloning: `python train_bc.py --config cartpole_bc.yaml`