# SPACESHIP TITANIC - PASSENGER TRANSPORTATION PREDICTION
Gilmar Neves

## Summary
- [Introduction](#introduction)
- [Tech and Tools](#techandtools)
    - [Architecture](#architecture)
- [Installation](#installation)
- [Use](#use)


## Introduction

Spaceship Titanic is a learning dataset from Kaggle, inspired by the famous Titanic problem and designed to help machine learning enthusiasts develop their first projects.

It features a synthetic dataset of passengers, complete with various attributes, and can be used to predict whether or not these passengers were transported to another dimension following the spaceship collision with a space-time anomaly.

This project aimed the creation of a ML model for predicting wether or not a passanger was transported based on some of the features available from each person.
Moreover, an ML app was developed for creating an interface to use such model with any user input.

The processing of EDA and ML model creating can be found in the notebooks folder in this project. It leads to a IPYNB file that can be visualized in tools like Jupyter and Google Colab.

The chosen model was
**SVM using {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}**
and its best results were an accuracy of about 75%.

The results were also submitted for Kaggle evaluation achieving 74% of accuracy.

## Tech and Tools
The project was solely developed in Python as a whole.
For the ML development, the librabry scikit-learn (https://scikit-learn.org/stable/) was the choice.
The user interface was created using Streamlit (https://streamlit.io/).
And the database is SQLite (https://www.sqlite.org/).

### Architecture

![app_arch](/images/app_arch.png)