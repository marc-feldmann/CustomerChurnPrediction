import pickle
import xgboost as xgb
import os
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.constraints import maxnorm
from keras.regularizers import l2
from keras.layers import GaussianNoise
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential
import keras.metrics
import tensorflow as tf
from xgboost import train
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import PowerTransformer, RobustScaler, MinMaxScaler
from sklearn.preprocessing import 
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV, StratifiedKFold)
import sklearn.metrics
import time
from matplotlib.lines import Line2D
from multiprocessing.sharedctypes import Value
from datetime import datetime
from imblearn.pipeline import Pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import numpy as np

np.set_printoptions(formatter={"float_kind": "{:f}".format})
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
pd.set_option("float_format", "{:f}".format)
pd.set_option('display.max_columns', None)
marker = "v3_HO_bundle1_grid1_"


# 1) Data Preprocessing
# 1a) Join Data and Labels
data = pd.read_table(
    "C:\\Users\\marc.feldmann\\Documents\\data_science_local\\CustomerChurnPrediction\\data\\orange_small_train.data"
)
data_labels = pd.read_table(
    "C:\\Users\\marc.feldmann\\Documents\\data_science_local\\CustomerChurnPrediction\\data\\orange_small_train_churn.labels",
    header=None,
    names=["Churn"],
)


# drop NaN-only columns and rows
# write-up: create charts: ordered columns/rows, number of NaNs
data.dropna(0, how='all', inplace=True) # dropt alle zeilen die nur Nas haben
data.dropna(1, inplace=True, thresh=data.shape[0] * 0.2) # dropt all spaltendie 

# 1b) Encode Categorical Features: Since Too Many Different Values, Create
# Dummy Variables Only For Most Frequent Column Values
data_obj_columns = data.select_dtypes(include=["object"]).columns.tolist()

# loop: for each obj columns, take the m most frequent values and save to
# list, apply get dummies to column for list values only, drop column
for iteration, clm in enumerate(data_obj_columns):
    print(
        "Encoding categorical variable ",
        iteration + 1,
        "/ ",
        len(data_obj_columns))
    most_freq_vals = data[clm].value_counts()[:20].index.tolist()
    dummy_clms = pd.get_dummies(
        data[clm].loc[data[clm].isin(most_freq_vals)], prefix=clm
    )
    data = pd.merge(
        data,
        dummy_clms,
        left_index=True,
        right_index=True,
        how="outer")
    for dum_clm in data[dummy_clms.columns]:
        data[dum_clm].fillna(0, inplace=True)
    data.drop(clm, axis=1, inplace=True)

# data.iloc[:, 174:].apply(pd.Series.value_counts)

# 1c) Since fact that values are NaN/missing itself might contain
# predictive power, create binary indicator column for each column with
# NaNs
for clm in data:
    if data[clm].isna().sum() > 0:
        print("Creating NaN indicator variable for column", clm)
        data.insert(data.shape[1], f"{clm}_NaNInd", 0)
        data[f"{clm}_NaNInd"] = np.where(np.isnan(data[clm]), 1, 0)

# 1d) Handle Missing Values: Mean Imputation
round(data.isna().sum().sum() / (data.shape[0] * data.shape[1]), 10)
# >>> data is sparse, NaNs in >16% of cells at this point

# ### mean imputation
# for iteration, clm in enumerate(data):
#      print('Imputing mean for NaNs in column ', iteration+1, '/ ', data.shape[1], '...')
#      data[clm].fillna(data[clm].mean(), inplace=True)

# median imputation
for iteration, clm in enumerate(data):
    print("Imputing median for NaNs in column ",
          iteration + 1, "/ ", data.shape[1], "...")
    data[clm].fillna(data[clm].median(), inplace=True)

# ### mode imputation
# for iteration, clm in enumerate(data):
#      print('Imputing mode for NaNs in column ', iteration+1, '/ ', data.shape[1], '...')
#      data[clm].fillna(data[clm].mode(), inplace=True)

# min imputation
# for iteration, clm in enumerate(data):
#      print('Imputing minimum for NaNs in column ', iteration+1, '/ ', data.shape[1], '...')
#      data[clm].fillna(data[clm].min(), inplace=True)

# ### max imputation
# for iteration, clm in enumerate(data):
#      print('Imputing maximum for NaNs in column ', iteration+1, '/ ', data.shape[1], '...')
#      data[clm].fillna(data[clm].max(), inplace=True)

# ### frequency imputation
# for iteration, clm in enumerate(data):
#      print('Imputing frequency for NaNs in column ', iteration+1, '/ ', data.shape[1], '...')
#      data[clm].fillna(data[clm].isna().sum(), inplace=True)

# ALTERNATIVE: kNN imputation - others' tests have suggested only minor contribution to model quality, but long runtime - so, decided to go with simple mean imputation
# imputer = KNNImputer(n_neighbors=2)
# imputer.fit_transform(data)

data.isna().sum().sum() == 0

# 1e) Split Data Into Training, Validation, and Test Data Sets
# since I do not have test data target values used in the KDD competition,
# will work only with subsets of competition's training data set (model
# training, evaluation)

# Check Class Distribution Prior to Splitting
data_labels["Churn"] = (data_labels["Churn"] + 1) / 2
# plt.hist(data_labels['Churn'], bins=3)
# plt.show()
# >>> Churn class is heavily imbalanced (as expected)

# feature selection
# data = SelectKBest(chi2, k=400).fit_transform(data, data_labels)

# Stratified splitting to create validation and test data
# data, data_labels = data.iloc[:, :], data_labels.iloc[:, :]
data_train, data_test, data_train_labels, data_test_labels = train_test_split(
    data, data_labels, test_size=0.2, stratify=data_labels
)
# data_test, data_val, data_test_labels, data_val_labels = train_test_split(data_test, data_test_labels, test_size=0.5, stratify=data_test_labels)

# Scale Data (Standardize/Normalize Data). Procedure: Fit scaler to train data, then apply fitted scaler to train and test data
### MinMaxScaler (Normalization)
scaler = MinMaxScaler()
data_train.iloc[:, :174] = scaler.fit_transform(data_train.iloc[:, :174])
data_test.iloc[:, :174] = scaler.transform(data_test.iloc[:, :174])

### RobustScaler (Standardization)
# scaler_robscal = RobustScaler()
# data_train.iloc[:, :174] = scaler.fit_transform(data_train.iloc[:, :174])
# data_test.iloc[:, :174] = scaler.transform(data_test.iloc[:, :174])

#### PowerTransformer (Standardization)
# scaler = PowerTransformer()
# data_train.iloc[:, :174] = scaler.fit_transform(data_train.iloc[:, :174])
# data_test.iloc[:, :174] = scaler.transform(data_test.iloc[:, :174])


# Reduce dataset via PCA
# pca = PCA(n_components=55)
# data_train = pca.fit_transform(data_train)
# data_test = pca.transform(data_test)


# 2) Model Building and Training: 'Deep' Neural Network Binary Classifier

# 2A) CREATE XGBOOST classifier ('Setting up the baseline model')
# legitimize choice of XGBoost:
# https://ieeexplore.ieee.org/abstract/document/7937698/authors#authors

# ## fit and evaluate on training data
# model_xgb = xgb.XGBClassifier()
# model_xgb.fit(data_train.iloc[:100, :], data_train_labels.iloc[:100, :])

# data_train_preds = model_xgb.predict(data_train)
# xgb_accuracy_train = sklearn.metrics.accuracy_score(data_train_labels, data_train_preds)
# xgb_recall_train = sklearn.metrics.precision_score(data_train_labels, data_train_preds)
# xgb_precision_train = sklearn.metrics.recall_score(data_train_labels, data_train_preds)

# ## generate and evaluate predictions on test data
# data_test_preds = model_xgb.predict(data_test)
# xgb_accuracy_test = sklearn.metrics.accuracy_score(data_test_labels, data_test_preds)
# xgb_recall_test = sklearn.metrics.recall_score(data_test_labels, data_test_preds)
# xgb_precision_test = sklearn.metrics.precision_score(data_test_labels, data_test_preds)

# plot confusion matrices
# ax = sns.heatmap(confusion_matrix(data_test_labels, data_test_preds), annot=True, fmt='g', cmap='Blues')
# ax.set_title('Confusion Matrix for Test Data');
# ax.set_xlabel('Customer action predicted by model')
# ax.set_ylabel('Actual customer action');
# ax.xaxis.set_ticklabels(['no churn','churn'])
# ax.yaxis.set_ticklabels(['no churn','churn'])
# plt.show()

# 2B) CREATE KERAS Feed-forward neural network CLASSIFICATION model to
# wrap into Scikit (required since we want to use scikits GridSearch
# class)

# GridSearch Optimization of Keras Classifier:

# review:
# (X) 1) variables properly implemented in net architecture
# (X) 2) variables correspond properly between net architecture and param_grids
# (X) 3) variables correspond properly across param_grids
# (X) 4) variables in model definition are set to default value / deactivated
# (X) 5) variable grid values are not complete nonsense

# specify MLP / FNN (feed-forward neural network model)
start_time = time.time()


def create_model(
    learning_rate=0.001,
    dropout_rate=0.0,
    noise=0.001,
    reg=0.0,
    beta_1=0.9,
    beta_2=0.999,
    weight_constraint=100.0,
    deep="n",
    neurons=120,
):
    model = Sequential()
    model.add(
        Dense(
            350,
            activation="relu",
            input_dim=data_train.shape[1],
            kernel_constraint=maxnorm(weight_constraint),
            activity_regularizer=l2(reg),
        )
    )
    model.add(Dropout(dropout_rate))
    model.add(GaussianNoise(stddev=noise))
    model.add(
        Dense(
            neurons,
            activation="relu",
            kernel_constraint=maxnorm(weight_constraint),
            activity_regularizer=l2(reg),
        )
    )
    model.add(Dropout(dropout_rate))
    model.add(GaussianNoise(stddev=noise))
    if deep == "y":
        model.add(
            Dense(
                round(neurons / 4 * 3, 0),
                activation="relu",
                kernel_constraint=maxnorm(weight_constraint),
                activity_regularizer=l2(reg),
            )
        )
        model.add(Dropout(dropout_rate))
        model.add(GaussianNoise(stddev=noise))
        model.add(
            Dense(
                round(neurons / 4 * 2, 0),
                activation="relu",
                kernel_constraint=maxnorm(weight_constraint),
                activity_regularizer=l2(reg),
            )
        )
        model.add(Dropout(dropout_rate))
        model.add(GaussianNoise(stddev=noise))
        model.add(
            Dense(
                round(neurons / 4 * 1, 0),
                activation="relu",
                kernel_constraint=maxnorm(weight_constraint),
                activity_regularizer=l2(reg),
            )
        )
        model.add(Dropout(dropout_rate))
        model.add(GaussianNoise(stddev=noise))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(
                name="ROC_AUC"),
            "accuracy",
            keras.metrics.Precision(
                name="precision"),
            keras.metrics.Recall(
                name="recall"),
        ],
    )
    return model


# wrap the created Keras model with Scikit's KerasClassifier
model = KerasClassifier(build_fn=create_model, verbose=2)

# define parameter set GridSearch should search (the 'grid')
grid_name = "param_bundle4_grid"

if grid_name == "param_bundle1_grid":
    param_grid = dict(
        batch_size=[80, 160, 240],
        learning_rate=[0.0001, 0.001, 0.01],
        epochs=[5, 20, 50],
    )
elif grid_name == "param_bundle2_grid":
    param_grid = dict(
        batch_size=[80],
        learning_rate=[0.0001],
        epochs=[5],
        dropout_rate=[0.1, 0.5, 0.9],
        noise=[0.001, 0.1, 1],
        reg=[0.001, 0.01, 0.5],
    )
elif grid_name == "param_bundle3_grid":
    param_grid = dict(
        batch_size=[80],
        learning_rate=[0.0001],
        epochs=[5],
        dropout_rate=[0.1],
        noise=[0.001],
        reg=[0.001],
        beta_1=[0.8, 0.9, 0.99],
        beta_2=[0.990, 0.995, 0.999],
        weight_constraint=[0.5, 2.0, 8.0],
    )
elif grid_name == "param_bundle4_grid":
    param_grid = dict(
        batch_size=[80],
        learning_rate=[0.0001],
        epochs=[5],
        dropout_rate=[0.1],
        noise=[0.001],
        reg=[0.001],
        beta_1=[0.8],
        beta_2=[0.999],
        weight_constraint=[0.5],
        deep=["n", "y"],
        neurons=[30, 180, 350],
    )
elif grid_name == "param_bundleTEST_grid":
    param_grid = dict(
        batch_size=[80],
        learning_rate=[0.001],
        epochs=[50],
        dropout_rate=[0.5],
        noise=[0.001],
        reg=[0.001],
        beta_1=[0.8],
        beta_2=[0.999],
        weight_constraint=[8.0],
        deep=["n"],
        neurons=[350],
    )

# fit GridSearch model
scoring = {
    "F1": sklearn.metrics.make_scorer(sklearn.metrics.f1_score),
    "ROC_AUC": sklearn.metrics.make_scorer(sklearn.metrics.roc_auc_score),
    "Accuracy": sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score),
    "Recall": sklearn.metrics.make_scorer(sklearn.metrics.recall_score),
    "Precision": sklearn.metrics.make_scorer(sklearn.metrics.precision_score),
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    verbose=3,
    refit="F1",
    n_jobs=2,
    scoring=scoring,
    return_train_score=True,
    cv=StratifiedKFold(n_splits=3, shuffle=True),
)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(data_train_labels.values),
    y=data_train_labels.values.reshape(-1),
)
class_weights = dict(zip(np.unique(data_train_labels.values), class_weights))
grid_result = grid.fit(
    data_train,
    data_train_labels,
    class_weight=class_weights)

# Save GridSearchCV Evaluation Results per Grid
print("k-fold validation for grid %s:" % grid_name)
print("Best: F1 (val) = %f using %s" %
      (grid_result.best_score_, grid_result.best_params_))
results = pd.DataFrame(grid_result.cv_results_["params"])
results["means_val_F1"] = grid_result.cv_results_["mean_test_F1"]
results["means_val_ROC_AUC"] = grid_result.cv_results_["mean_test_ROC_AUC"]
results["means_val_Accuracy"] = grid_result.cv_results_["mean_test_Accuracy"]
results["means_val_Recall"] = grid_result.cv_results_["mean_test_Recall"]
results["means_val_Precision"] = grid_result.cv_results_["mean_test_Precision"]
results["means_train_F1"] = grid_result.cv_results_["mean_train_F1"]
results["means_train_ROC_AUC"] = grid_result.cv_results_["mean_train_ROC_AUC"]
results["means_train_Accuracy"] = grid_result.cv_results_[
    "mean_train_Accuracy"]
results["means_train_Recall"] = grid_result.cv_results_["mean_train_Recall"]
results["means_train_Precision"] = grid_result.cv_results_[
    "mean_train_Precision"]
filename = ("FNN_clf_GRResults_" + grid_name +
            datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))
results.to_csv(
    "C:\\Users\\marc.feldmann\\Documents\\data_science_local\\OCC\\results\\\\hyparam_opt\\"
    + filename
    + ".csv",
    decimal=",",
)

print("TOTAL RUNTIME: --- %s seconds ---" % (time.time() - start_time))

# 3) Model Evaluation on unseen Data (the priorly held out "data_test" set)
# ## specify Feed-forward neural network model with the Grid search optimized parameters
#

def create_final_model(
    learning_rate=0.0001,
    dropout_rate=0.1,
    noise=0.001,
    reg=0.001,
    beta_1=0.8,
    beta_2=0.999,
    weight_constraint=0.5,
    neurons=180,
):
    model = Sequential()
    model.add(
        Dense(
            350,
            activation="relu",
            input_dim=data_train.shape[1],
            kernel_constraint=maxnorm(weight_constraint),
            activity_regularizer=l2(reg),
        )
    )
    model.add(Dropout(dropout_rate))
    model.add(GaussianNoise(stddev=noise))
    model.add(
        Dense(
            neurons,
            activation="relu",
            kernel_constraint=maxnorm(weight_constraint),
            activity_regularizer=l2(reg),
        )
    )
    model.add(Dropout(dropout_rate))
    model.add(GaussianNoise(stddev=noise))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(
                name="ROC_AUC"),
            "accuracy",
            keras.metrics.Precision(
                name="precision"),
            keras.metrics.Recall(
                name="recall"),
        ],
    )
    return model


# wrap the created Keras model with Scikit's KerasClassifier
model = KerasClassifier(build_fn=create_final_model, verbose=1)

# fit and evaluate GridSearch-optimized model on all training data (no
# kfold cross validation)
model.fit(
    data_train,
    data_train_labels,
    batch_size=80,
    epochs=50,
    verbose=2,
    class_weight=class_weights,
)

# generate predictions on test data
data_test_preds = model.predict(data_test)

# Confusion matrices
ax2 = sns.heatmap(
    confusion_matrix(data_test_labels, data_test_preds),
    annot=True,
    fmt="g",
    cmap="Blues",
)
ax2.set_title("Confusion Matrix on Test Data")
ax2.set_xlabel("Customer action predicted by model")
ax2.set_ylabel("Actual customer action")
ax2.xaxis.set_ticklabels(["no churn", "churn"])
ax2.yaxis.set_ticklabels(["no churn", "churn"])
filename = ("FNN_clf_confmat_" + grid_name +
            datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))
plt.savefig(
    "C:\\Users\\marc.feldmann\\Documents\\data_science_local\\OCC\\results\\"
    + filename
    + ".png"
)
plt.close()

sklearn.metrics.accuracy_score(data_test_preds, data_test_labels)
sklearn.metrics.precision_score(data_test_preds, data_test_labels)
sklearn.metrics.recall_score(data_test_preds, data_test_labels)
sklearn.metrics.f1_score(data_test_preds, data_test_labels)
sklearn.metrics.roc_auc_score(data_test_preds, data_test_labels)


pkl_filename = "optimized_model.pkl"
with open(pkl_filename, "wb") as file:
    pickle.dump(model, file)
