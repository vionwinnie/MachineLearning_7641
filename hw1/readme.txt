This folder contains three code files and two datasets used for classification. 
The codes on this folder runs on Python 3.5
Main packages used are scikit-learn and keras (for neural network)


Algorithm explored are k-nearest neighbor, decision trees, 
boosting (Adaboost and Gradient Boosting),neural network, and SVM (with different kernels)

------------------------------------------------------
Datasets

1. Wine Quality - White Wine

Link to Heart Disease Dataset: http://archive.ics.uci.edu/ml/datasets/Wine+Quality
The dataset is related to white variants of the Portuguese ”Vinho Verde” wine. It contains physicochemical
inputs: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur
dioxide, density, pH, sulphates, and alcohol with quality as the output.

2. Forest Cover Type

Link to Forest Cover Types Dataset: https://www.kaggle.com/uciml/forest-cover-type-dataset
This dataset contains tree observations from four areas of the Roosevelt National Forest in Colorado. All
observations are cartographic variables from 30 meter x 30 meter sections of forest. This dataset includes
information on tree type, shadow coverage, distance to nearby landmarks (roads etcetera), soil type, and
local topography. The output is type of trees and it is divided into 7 classes,

------------------------------------------------------
Code 

- utils.py : contains functions to preprocess datasets and drawing learning curve
- WineQuality_AllCodeCollection.ipynb: Jupyter notebook that explores the 5 algorithms on Wine Quality Dataset, with dependecies on utils.py
- ForestCover_AllCodeCollection.ipynb: Jupyter notebook that explores the 5 algorithms on Forest Cover Type Dataset, with dependecies on utils.py
