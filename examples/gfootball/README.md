This is the guidance for [Google Research Football](https://github.com/google-research/football).

### Installation

- `pip install gfootball`
- `pip install tizero`
- test the installation by `python3 -m gfootball.play_game --action_set=full`.

### Evaluate JiDi submissions locally

If you want to evaluate your JiDi submissions locally, please try to use tizero as illustrated [here](foothttps://github.com/OpenRL-Lab/TiZero#evaluate-jidi-submissions-locally).


### Convert dump file to video

After the installation, you can use tizero to convert a dump file to a video file.
The usage is `tizero dump2video <dump_file> <output_dir> --episode_length <the length> --render_type <2d/3d>`.

You can download an example dump file from [here](http://jidiai.cn/daily_6484285/daily_6484285.dump). 
And then execute `tizero dump2video daily_6484285.dump ./` in your terminal. By default, the episode length is 3000 and the render type is 2d.
Wait a minute, you will get a video file named `daily_6484285.avi` in your current directory.