# Spotify Popularity Prediction with Mamba Model
Predicting a song's popularity based on the most popular songs on Spotify using Mamba architecure and Spotify Audio features dataset.

![spotify](https://github.com/Rebeccaazoulay/DeepLearning_04621/assets/102752965/db7feed7-7c79-4d8f-9010-470095dd2196)

Project Explanation Video (English): https://www.youtube.com/watch?v=d8VXnhX3pAo

## Table of Contents
  * [Background](#background)
  * [Mamba](#mamba)
  * [Dataset](#dataset)
  * [Results](#results)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Usage](#usage)
  * [References](#references)



## Background
Predicting the popularity of songs is crucial for businesses aiming to remain competitive in the ever-expanding music industry. Initial attempts to predict popularity using basic machine learning techniques, including linear regression and logistic regression, yielded modest results. Our project explores the various factors influencing song popularity by utilizing a dataset comprising audio features and metadata for 4,771 tracks spanning the last 50 years. Leveraging the Mamba architecture, a deep learning model customized for long sequence modeling, our project aimed to enhance prediction accuracy. Surprisingly, the Mamba architecture yielded similar results to linear regression, prompting further investigation.

## Mamba
The architecture of our model is based on the Mamba model, which is a sequence model that can handle long data sequences. It builds on the Structured State Space (S4) model and introduces enhancements, particularly in time-variant operations. Mamba’s design includes a unique selection mechanism that adapts SSM parameters based on the input, allowing it to focus on relevant information within sequences.

The model transitions from a time-invariant to a time-varying framework, which impacts both the computation and efficiency of the system. To address the computational challenges introduced by this time-variance, Mamba employs a hardware-aware algorithm. This algorithm enables efficient computation on modern hardware, like GPUs, by using techniques such as kernel fusion, parallel scan, and recomputation.

Mamba also simplifies its architecture by integrating the SSM design with MLP blocks, resulting in a homogeneous and streamlined structure. This enhances the model’s capability for general sequence modeling across various data types, including language, audio, and genomics, while maintaining efficiency in both training and inference.

## Dataset
For this project, we constructed a comprehensive song dataset that contains a plethora of feature types for 4,771 tracks from the last 50 years, which includes various audio features of Spotify songs such as acousticness, danceability, energy, tempo, etc. Here are some example songs from the dataset, without any preprocessing:

| artist_name   |  track_id               |track_name    | acousticness | danceability | duration_ms | energy | instrumentalness | key | liveness | loudness | mode  | speechiness | tempo   | time_signature | valence | popularity   |
|:-------------:|:-----------------------:|:-----------:|:------------:|:------------:|:-----------:|:------:|:----------------:|:---:|:--------:|:--------:|:-----:|:-----------:|:-------:|:--------------:|:-------:|:----------:|
| Ariana Grande  | 5D34wRmbFS29AjtTOP2QJe |   yes, and?   |    0.194     |    0.785     | 214994      | 0.766  | 7        |  1  |  0.107   | -6.551   |   1   |   0.0503    | 119.029 |        4       | 0.804   |    84      |
| Mitski  | 3vkCueOmm7xQDoJ17W1Pm3 |   My Love Mine All Mine   |   0.868    |    0.504     | 137773      | 0.308  | 0.135  |  9  |  0.158   | -14.958   |   1   |   0.0321    | 113.95 |        4       | 0.121   |    96      |
| Feid  | 7bywjHOc0wSjGGbj04XbVi |   LUNA   |   0.131   |    0.774     | 196800      | 0.86  | 0  |  7  |  0.116   | -2.888   |   0   |   0.13    | 100.019 |        4       | 0.446   |    95      |

We performed normalization procedures and fed the data to two models, linear regression and mamba based, with the goal of predicting the popularity of the given songs.


## Results
Using the Mamba architecture on song’s popularity prediction, we essentially got equivalent results to those obtained with the linear regression approach.

## Prerequisites
- Python
- PyTorch
- scikit-learn
- matplotlib
- mamba_ssm
- pandas
- numpy
- GPU

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

This script trains the Mamba model on the Spotify dataset, performs hyperparameter tuning, and evaluates the model's performance.

## References
* https://github.com/state-spaces/mamba
* https://cs230.stanford.edu/projects_fall_2020/reports/55822810.pdf
* https://cs229.stanford.edu/proj2015/140_report.pdf
* https://github.com/MattD82/Predicting-Spotify-Song-Popularity/blob/master/README.md
* https://github.com/twillstw/Spotify-Popularity-Prediction/tree/master
* https://towardsdatascience.com/song-popularity-predictor-1ef69735e380
  

* https://github.com/twillstw/Spotify-Popularity-Prediction/tree/master
* https://towardsdatascience.com/song-popularity-predictor-1ef69735e380
  
