# Car Evaluation System Using SVM

## Description

This project implements a Car Evaluation System using the Support Vector Machine (SVM) algorithm. The system classifies cars into different categories based on features such as buying price, maintenance cost, number of doors, seating capacity, luggage boot size, and safety level.

The model is trained using the UCI Car Evaluation Dataset and predicts whether a car is Unacceptable, Acceptable, Good, or Very Good.

## Features

* Data loading and preprocessing
* Exploratory Data Analysis (EDA)
* Label Encoding of categorical data
* Feature Scaling using StandardScaler
* SVM Model Training
* Hyperparameter Tuning using GridSearchCV
* Model Evaluation using Accuracy Score and Classification Report
* Confusion Matrix Visualization
* SVM Kernel Comparison
* Car Category Prediction

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Dataset

UCI Car Evaluation Dataset

Attributes:

* Buying Price
* Maintenance Cost
* Doors
* Persons Capacity
* Luggage Boot Size
* Safety

Target Classes:

* Unacceptable (unacc)
* Acceptable (acc)
* Good (good)
* Very Good (vgood)

## How to Run

1. Install required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn

2. Run the program:

python car_evaluation_svm.py

## Output

The project generates:

* EDA Plots
* Confusion Matrix
* Kernel Comparison Graph
* Classification Report
* Accuracy Scores
* Car Evaluation Predictions

## Sample Prediction

Input:

* Buying = low
* Maintenance = low
* Doors = 4
* Persons = more
* Luggage Boot = big
* Safety = high

Output:

* Predicted Class = VGOOD

## Objective

To build an accurate machine learning model that evaluates and classifies cars based on their characteristics using the Support Vector Machine (SVM) algorithm.

## Author

Machine Learning Project – Car Evaluation System Using SVM
