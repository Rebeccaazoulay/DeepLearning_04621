# DeepLearning_04621
Predicting a song's popularity based on the most popular songs on Spotify using Mamba architecure

Based on the paper:

Nir Levine, Tom Zahavy, Daniel J. Mankowitz, Aviv Tamar, Shie Mannor [Shallow Updates for Deep Reinforcement Learning](https://cs230.stanford.edu/files_winter_2018/projects/6970963.pdf)

Video:

[YouTube](https://youtu.be/i8Cnas7QrMc) - https://youtu.be/i8Cnas7QrMc

![spotify](https://github.com/Rebeccaazoulay/DeepLearning_04621/assets/102752965/db7feed7-7c79-4d8f-9010-470095dd2196)


Github Link : [https://github.com/taldatech/pytorch-ls-ddpg](https://github.com/Rebeccaazoulay/DeepLearning_04621)

  * [Background](#background)
  * [Prerequisites](#prerequisites)
  * [Files in the repository](#files-in-the-repository)
  * [API (`ls_dqn_main.py --help`)](#api---ls-dqn-mainpy---help--)
  * [Playing](#playing)
  * [Training](#training)
  * [Playing Atari on Windows](#playing-atari-on-windows)
  * [TensorBoard](#tensorboard)
  * [References](#references)

## Background
Predicting song popularity is particularly important in keeping businesses competitive within a growing music industry. But what exactly makes a song popular? Starting with
a big Song Dataset, a collection of audio features and metadata for approximately 5000 songs, we implemented a Mamba architecure to predict popularity and determined the types of features that hold the most predictive power. We wanted to compare the results to the one obtained with different classification and regression algorithms.


## Prerequisites
|Library         | Version |
|----------------------|----|
|`Python`|  `3.5.5 (Anaconda)`|
|`torch`|  `0.4.1`|
|`tensorboard`|  `1.12.0`|


## Files in the repository

|File name         | Purpsoe |
|----------------------|------|
|`ls_dqn_main.py`| general purpose main application for training/playing a LS-DQN agent|
|`pong_ls_dqn.py`| main application tailored for Atari's Pong|
|`boxing_ls_dqn.py`| main application tailored for Atari's Boxing|
|`dqn_play.py`| sample code for playing a game, also in `ls_dqn_main.py`|


## API (`ls_dqn_main.py --help`)


You should use the `ls_dqn_main.py` file with the following arguments:

|Argument                 | Description                                 |
|-------------------------|---------------------------------------------|
|-h, --help       | shows arguments description             |
|-t, --train     | train or continue training an agent  |
|-p, --play    | play the environment using an a pretrained agent |
|-n, --name       | model name, for saving and loading |
|-k, --lsdqn	| use LS-DQN (apply LS-UPDATE every N_DRL), default: false |
|-j, --boosting| use Boosted-FQI as SRL algorithm, default: false |


## Training

Examples:

* `python ls_dqn_main.py --train --lsdqn -e boxing -l 10 -b 64`
* `python ls_dqn_main.py --train --lsdqn --boosting --dueling -m -e boxing -l 1000 -b 64`

For full description of the flags, see the full API.

## TensorBoard

TensorBoard logs are written dynamically during the runs, and it possible to observe the training progress using the graphs. In order to open TensoBoard, navigate to the source directory of the project and in the terminal/cmd:

`tensorboard --logdir=./runs`

* make sure you have the correct environment activated (`conda activate env-name`) and that you have `tensorboard`, `tensorboardX` installed.

## References
* https://cs230.stanford.edu/projects_fall_2020/reports/55822810.pdf
* https://cs229.stanford.edu/proj2015/140_report.pdf
* https://github.com/MattD82/Predicting-Spotify-Song-Popularity/blob/master/README.md}
* https://github.com/twillstw/Spotify-Popularity-Prediction/tree/master}
* https://towardsdatascience.com/song-popularity-predictor-1ef69735e380}
  
