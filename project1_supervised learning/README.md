# Supervised Learning Project: Social Media Comment Sentiment Dataset

## 1. Project Overview
This repository contains a self-collected dataset and related source code for a supervised learning project on sentiment classification.  
The goal of this project is to collect social media comments, organize and label them, clean the data, merge multiple sources, and use the final dataset to train a machine learning model for sentiment analysis.

This dataset was not directly taken from an existing public dataset. It was built through data collection, initial labeling, manual review, data cleaning, and dataset merging.


## 2. Dataset Description
The dataset in this project is mainly composed of comments collected from two social media platforms:

- YouTube
- Facebook

The collected comments were processed through multiple stages:
1. Raw comment collection
2. Initial labeling
3. Manual review and correction
4. Data cleaning and reorganization
5. Merging into one final dataset

The final dataset is intended for supervised learning tasks such as text classification and sentiment analysis.

## 3. Repository Structure
The recommended structure of this repository is as follows:

```text
project1_supervised_learning/
├── README.md
├── data/
│   ├── yt_original_comments.csv
│   ├── yt_label.csv
│   ├── yt_label_clean.csv
│   ├── fb_original_comments.csv
│   ├── fb_label.csv
│   ├── fb_label_clean.csv
│   └── Final_Dataset_Combined.csv
└── code/
    ├── yt_crawler.py
    ├── fb_crawler.py
    ├── label.py
    ├── data_clean.py
    ├── merge.py
    ├── train.py
    └── chart.py

## 4. Dataset Files

The repository includes the following dataset files:

yt_original_comments.csv
Raw comments collected from YouTube.
yt_label.csv
Initially labeled YouTube comments.
yt_label_clean.csv
Cleaned and reorganized YouTube comments after labeling.
fb_original_comments.csv
Raw comments collected from Facebook.
fb_label.csv
Initially labeled Facebook comments.
fb_label_clean.csv
Cleaned and reorganized Facebook comments after labeling.
Final_Dataset_Combined.csv
The final combined dataset containing cleaned comments from both YouTube and Facebook.
5. Column Description

The columns may vary slightly across different CSV files, but the main fields include:

comment
The text content of the social media comment.
label
The sentiment label assigned to the comment.
Other optional columns
Depending on the file, additional columns may include identifiers, platform source, or other metadata.
6. Label Definition

This project uses sentiment labels for classification.

1 = Positive
0 = Negative

If more label categories are added in the future, they should be defined accordingly.

7. Data Processing Workflow

The overall workflow of this project is as follows:

Collect comments from YouTube and Facebook
Perform initial labeling on the collected comments
Manually review and revise the labels
Remove empty, invalid, or duplicate entries
Clean noisy text and reorganize the data
Standardize the format of the datasets
Merge cleaned datasets from different platforms
Use the final dataset for model training and analysis
8. Code Description

This repository also includes source code used for data collection, preprocessing, training, and visualization.

yt_crawler.py
Crawls and collects YouTube comments.
fb_crawler.py
Crawls and collects Facebook comments.
label.py
Performs initial labeling on the collected comments.
After this step, the labels are manually reviewed and corrected.
data_clean.py
Cleans the data, reorganizes the dataset structure, and removes unnecessary or noisy entries.
merge.py
Merges the cleaned YouTube and Facebook datasets into one final dataset.
train.py
Trains the supervised learning model using the final combined dataset.
chart.py
Generates charts for data distribution and related statistical visualization.
9. Purpose of the Dataset

This dataset was created for a course project in supervised learning.
Its main purpose is to support experiments in sentiment classification on social media comments.

Possible tasks include:

Text classification
Sentiment analysis
Data distribution analysis
Model training and evaluation
10. Notes
This dataset was created for educational and course project purposes.
The repository includes both the dataset and the code used to process and analyze it.
The labeling process includes both initial labeling and manual review to improve data quality.

```md
The dataset was created specifically to satisfy the course requirement of building a dataset from scratch.