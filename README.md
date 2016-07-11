# Santander-Customer-Satisfaction

We built data models to predict if a customer is satisfied or dissatisfied with their banking experience at Santander Bank. The raw dataset and problem statement is taken from Santander Customer Satisfaction competition on Kaggle.

In the given data, a lot of it was missing or zero. This meant that we needed to select only those features which are important and which would help us make a good prediction. So the first step we took in feature engineering was to remove all those features which had zero variance, because all such features don’t really provide us with any useful data about the customer. Second, we found all the features that had duplicates, i.e., identical columns. We eliminated all the duplicates and maintained only one of the features.

XGBoost was our best model which gave the highest score of 0.842629 on the public leaderboard. We used the Python implementation of this library for our project. This was only gradient boosting library we found which supported handling missing values. It handled missing values by automati- cally finding its best imputation value based on the reduction on training loss. We believe that this was the deciding factor in this model being the best because a lot of training data had missing values and this model was able to deal with it really well.

