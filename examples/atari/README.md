## Installation

`pip install "gymnasium[atari]"`

Then install auto-rom via:
`pip install "gymnasium[accept-rom-license]"`

or:
```shell
pip install autorom
AutoROM --accept-license
```

or, if you can not download the ROMs, you can download them manually from [Google Drive](https://drive.google.com/file/d/1agerLX3fP2YqUCcAkMF7v_ZtABAOhlA7/view?usp=sharing).
Then, you can install the ROMs via:
```shell
pip install autorom
AutoROM --source-file <path-to-Roms.tar.gz>
````


## Usage

```shell
python train_ppo.py
```