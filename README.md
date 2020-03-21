# Recommendation System.
The goal for this project to get familiarize with different type of Collaborative Filtering Recommendation System
1. Model-Based CF Recommendation System.
    Trained the Recommendation Model using Alternating Mean Squares and evaluated the Testing Data on this trained model.
    RMSE Achieved: 1.23
2. User-Based CF Recommendation System.
    This algorithm produces a rating for a Business by a User by combining ratings of similar Users using the Pearson Correlation distance metric.
    RMSE Achieved: 1.10
3. Item-Based CF Recommendation System.
    This algorithm produces a rating for a Business by a User by combining ratings of similar Businesses for which User has given a rating.
    RMSE Achieved: 1.06

## Input 
spark-submit recommendationSystem.py TRAININGFILE TESTFILE NUMBER OUTPUTFILE

NUMBER denotes the caseid
1 - Model Based
2 - User Based
3 - Item Based 

## Output 
CSV file containing all prediction Result.

