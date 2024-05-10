# Spaceship Titanic
# Overview
The project inspects Stroke Prediction Dataset from Kaggle. 

The primary objectives are to clean the data, perform exploratory data analysis, statistical analysis, and apply various machine learning models for target variable Transported. Spaceship Titanic is ment for Kaggle competition.

## Dataset
Dataset can be downloaded from [Kaggle](https://www.kaggle.com/competitions/spaceship-titanic/overview).

## Python Libraries

This analysis was conducted using Python 3.11. The following packages were used:
- numpy=1.26.4
- pandas=2.2.1
- seaborn=0.13.2
- matplotlib=3.8.4
- scikit-learn=1.4.2
- shap=0.45.1
- lazypredict=0.2.12
- libxgboost=2.0.3
- lightgbm=4.3.0



## Findings

* Exploratory Data Analysis (EDA): Dataset has 8693 observations and 14 features in training set and 4277 observations in test set. Data is collected from passengers of Spaceship Titanic. Analasis showed that Passenger Deck depends on HomePlanned, while lower decks have most passengers. Some passengers get around to amenities, while some sleep in CryoSleep. Target demographic is mid 20s, but are in ranges of infant to 79 years old.
* Correlation: No 2 features are strong correlated, no feature pair with linear relationship. But some features are derivatives from one-another.
* Feature Engineering: 'Spending per person' was created, empty values imputed with mean and mode for Continues and Categorical features. 
* Statistical Testing:  Examining data showed that VIP passengers tend to be in their 30s. VIP passengers tend to spend more on overall amenities.
* Models: Various machine learning models (KNN, Support Vector Machines, Decision Tree, Random Forest, Naive Bayers, Gradient Boosting, LightXGB) were tested, as well as Hard Voting Classifier. Target feature was balanced
* Best model: Best model was 'LightXGB' with accuracy of 79%.



## Future Work

- Employing dimensionality reduction techniques like Principal Component Analysis (PCA) or t-Distributed Stochastic Neighbor Embedding (t-SNE) to condense the feature space and enhance interpretability.
