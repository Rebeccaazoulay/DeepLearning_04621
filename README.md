# DeepLearning_04621
Predicting a song's popularity based on the most popular songs on Spotify using Mamba architecure

Based on the paper:

James Pham, Edric Kyauk, Edwin Park [Predicting Song Popularity](https://cs230.stanford.edu/files_winter_2018/projects/6970963.pdf)

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
  * [TensorBoard](#tensorboard)
  * [References](#references)

## Background
Predicting song popularity is particularly important in keeping businesses competitive within a growing music industry. But what exactly makes a song popular? Starting with
a big Song Dataset, a collection of audio features and metadata for approximately 5000 songs, we implemented a Mamba architecure to predict popularity and determined the types of features that hold the most predictive power. We wanted to compare the results to the one obtained with different classification and regression algorithms.

## Prerequisites

- Python 3.x
- PyTorch
- scikit-learn
- matplotlib
- mamba_ssm
- pandas
- numpy
- optuna

## Files in the repository

|File name         | Purpsoe |
|----------------------|------|
|`ls_dqn_main.py`| general purpose main application for training/playing a LS-DQN agent|
|`pong_ls_dqn.py`| main application tailored for Atari's Pong|
|`boxing_ls_dqn.py`| main application tailored for Atari's Boxing|
|`dqn_play.py`| sample code for playing a game, also in `ls_dqn_main.py`|


# Spotify Popularity Prediction with Mamba Model

This repository contains code for training and evaluating a popularity prediction model for Spotify songs using the Mamba model architecture. The dataset used is the Spotify Audio Features dataset from November 2018.

## Installation

To install the necessary dependencies, run:
pip install mamba_ssm

## Usage

1. Clone the repository:
git clone https://github.com/your_username/spotify-popularity-prediction.git

2. Navigate to the project directory:
cd spotify-popularity-prediction

3. Run the provided Python script:
python train_model.py

This script trains the Mamba model on the Spotify dataset, performs hyperparameter tuning using Optuna, and evaluates the model's performance.

## Dataset

The dataset used for this project is the [Spotify Audio Features dataset](https://www.kaggle.com/nadintamer/top-tracks-of-2017), which includes various audio features of Spotify songs such as acousticness, danceability, energy, tempo, etc.

## References
* https://cs230.stanford.edu/projects_fall_2020/reports/55822810.pdf
* https://cs229.stanford.edu/proj2015/140_report.pdf
* https://github.com/MattD82/Predicting-Spotify-Song-Popularity/blob/master/README.md
* https://github.com/twillstw/Spotify-Popularity-Prediction/tree/master
* https://towardsdatascience.com/song-popularity-predictor-1ef69735e380
  
