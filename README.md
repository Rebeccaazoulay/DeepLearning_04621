# Spotify Popularity Prediction with Mamba Model
Predicting a song's popularity based on the most popular songs on Spotify using Mamba architecure. The dataset used is a Spotify Audio Features dataset.

Based on the paper:

James Pham, Edric Kyauk, Edwin Park [Predicting Song Popularity](https://cs230.stanford.edu/files_winter_2018/projects/6970963.pdf)

Video:

[YouTube](https://youtu.be/i8Cnas7QrMc) - https://youtu.be/i8Cnas7QrMc

![spotify](https://github.com/Rebeccaazoulay/DeepLearning_04621/assets/102752965/db7feed7-7c79-4d8f-9010-470095dd2196)


Github Link : (https://github.com/Rebeccaazoulay/DeepLearning_04621)

  * [Background](#background)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Usage](#usage)
  * [Dataset](#dataset)
  * [References](#references)

## Background
Predicting the popularity of songs is crucial for businesses aiming to remain competitive
in the ever-expanding music industry. Initial attempts to predict popularity using basic
machine learning techniques, including linear regression and logistic regression, yielded
modest results. Our project explores the various factors influencing song popularity by
utilizing a dataset comprising audio features and metadata for 4,771 tracks spanning the
last 50 years. Leveraging the Mamba architecture, a deep learning model customized for
long sequence modeling, our project aimed to enhance prediction accuracy. Surprisingly,
the Mamba architecture yielded similar results to linear regression, prompting further
investigation.

## Prerequisites

- Python 3.x
- PyTorch
- scikit-learn
- matplotlib
- mamba_ssm
- pandas
- numpy

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

The dataset used for this project is the [Spotify Audio Features dataset], made of 4,471 songs which includes various audio features of Spotify songs such as acousticness, danceability, energy, tempo, etc.

## References
* https://cs230.stanford.edu/projects_fall_2020/reports/55822810.pdf
* https://cs229.stanford.edu/proj2015/140_report.pdf
* https://github.com/MattD82/Predicting-Spotify-Song-Popularity/blob/master/README.md
* https://github.com/twillstw/Spotify-Popularity-Prediction/tree/master
* https://towardsdatascience.com/song-popularity-predictor-1ef69735e380
  
