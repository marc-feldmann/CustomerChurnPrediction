***main file: report.ipynb***

# Preventing Customer Churn with Feedforward Neural Networks

*Disclaimer: The main file "report.ipynb" is a mock project report that serves educational purposes only. The company data is public ([https://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data](https://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data)). All other company information is fictional. The author has no commercial relationship with mentioned parties.*


### Executive Summary

Customer retention is critical for French telecommunication provider Orange because retaining customers prevents revenue losses and replacement costs. However, Orange lacks an automated, scalable, and data-driven method for predicting customer churn that would allow Orange to initiate retention measures before customers leave. At the moment, predicting customer churn at Orange relies more or less on sporadic guessing. To address this problem, Orange requested a proof-of-concept for a "deep learning" articial neural networks that can help identify customers who will likely churn so that timely retention measures can be initiated.

As a main result, the conducted proof of concept suggests Orange should not rely solely on deep ANN models for predicting churn at scale, but combine ANNs with gradient boosted trees and other model classes in an ensemble approach. In any case, a comprehensive yet precise identification of churning customers seems only possible with more comprehensive and consistent methods for collecting customers’behavioral data.


### Main requirements

sklearn, keras/tf, matplotlib/seaborn


### Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


### Roadmap

Data-related optimization potentials:

1. Focus more systematically on feature engineering and/or selection (e.g. overproduce-and-select approach).
2. Improve data situation to include additional features and/or observations.
3. Use PCA or SVD to create smaller number of features with more predictive power; thus also address curse of dimensionality and further mitigate ANNs' overfitting tendency.
4. Cluster observations and impute missing values (NaN) with cluster-means/medians/etc..
5. Use outlier-insensitive scaler (e.g. sklearn's 'RobustScaler') or remove outliers.
6. Try over/undersampling techniques (e.g., SMOTE, SMOTETomek) instead of class weights to address imbalance of the target variable.
7. Perform infrequent category replacement for test and train sets separately.

Model-related optimization potentials:

1. Implement and compare performance of other model classes, especially (optimized) XGBoost, (optimized) Log Regression models to rule out that model performance is already at the upper boundary of what's possible with this dataset.
2. Try ensemble methods (e.g., boosting, bagging) to stack and combine votes of multiple models from same model class or multiple models from multiple model classes.
3. Optimize network architecture (e.g., long short-term memory).
4. With better computational resources, un-stage grid search and optimize all hyperparameters at once to less easily oversee feature interdependencies.
5. Significantly expand grid search ranges (only narrow ranges covered in current grids) to avoid getting stuck in local optima.
6. Identify and remove multicollineate features to decrease model complexity and overfitting risk
