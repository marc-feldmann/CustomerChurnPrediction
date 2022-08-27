'''
This version, v6, reflects the major script revision in August.
It will be the basis for revising the Jupyter-based "Project Report".
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import seaborn as sns
from keras.models import Sequential 
from keras.layers import Dense, Activation 
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense, Input, Embedding, Lambda, Dropout, Activation, Reshape, GlobalAveragePooling1D, merge, Flatten, Bidirectional, CuDNNGRU, add, Conv1D, GlobalMaxPooling1D,MaxPooling1D
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from numpy.random import seed
import tensorflow as tf
seed(3992)
tf.random.set_seed(3992)


##### SPLIT DATA
X = pd.read_table('data/orange_small_train.data')
y = pd.read_table('data/orange_small_train_churn.labels', header = None,sep='\t').loc[:, 0].astype('category')

X.dropna(axis=0, how='all', inplace=True)
X.dropna(axis=1, how='all', inplace=True)

X.info(verbose=True)
X['Var73']=X['Var73'].astype('float')


# Replace all infrequent cat values with same value
features_cat = list(X.select_dtypes(include=['object']).columns)
for feat in features_cat:
    X.loc[X[feat].value_counts(dropna=False)[X[feat]].values < X.shape[0] * 0.02, feat] = 'RARE_VALUE'
X[X == 'RARE_VALUE'].count().sum()


# y.value_counts().plot.bar()
# plt.ylabel('value')
# plt.title('churn value for each class')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=3992)

##################################################################################################
##### FIT AND APPLY DATA PREP ON TRAINING SET
# (later:
# - include NaN indicator clm generation
# - fit infrequent value replacement to train, apply to test - probably with dictionary/loop)

# imputation, encoding, scaling


# insert NaN indicator columns
for clm in X_train:
    if X_train[clm].isna().sum() > 0:
        X_train.insert(X_train.shape[1], f"{clm}_NaNInd", 0)
        X_train[f"{clm}_NaNInd"] = np.where(pd.isnull(X_train[clm]), 1, 0)


# missing value imputation: fit and apply imputers
from sklearn.impute import SimpleImputer

features_num_train = list(X_train.select_dtypes(include=['float']).columns)
imputer_nums = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train[features_num_train] = imputer_nums.fit_transform(X_train[features_num_train])

features_cat_train = list(X_train.select_dtypes(include=['object']).columns)
imputer_cats = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='unknown')
X_train[features_cat_train] = imputer_cats.fit_transform(X_train[features_cat_train])

X_train.isna().sum().sum()


# Encode categorical features
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
enc = make_column_transformer((OneHotEncoder(max_categories=20, handle_unknown='ignore'), features_cat_train), remainder='passthrough')
transformed = enc.fit_transform(X_train)
enc_df = pd.DataFrame(transformed, columns=enc.get_feature_names())
cols = enc_df.columns.tolist()
cols = cols[168:] + cols[:168]
X_train = enc_df[cols]
X_train.info(verbose=True)


# fit and apply scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[features_num_train] = scaler.fit_transform(X_train[features_num_train])


# transform target variable
y_train.replace(-1, 0, inplace=True)

X_train.shape
X_test.shape

##################################################################################################
##### APPLY DATA PREP TEST SET


# insert NaN indicator columns
for clm in X_test:
    if X_test[clm].isna().sum() > 0:
        X_test.insert(X_test.shape[1], f"{clm}_NaNInd", 0)
        X_test[f"{clm}_NaNInd"] = np.where(pd.isnull(X_test[clm]), 1, 0)


# missing value imputation: apply imputers
X_test[features_num_train] = imputer_nums.transform(X_test[features_num_train])
X_test[features_cat_train] = imputer_cats.transform(X_test[features_cat_train])
X_test.isna().sum().sum()


# Encode categorical features
transformed = enc.transform(X_test)
enc_df = pd.DataFrame(transformed, columns=enc.get_feature_names())
cols = enc_df.columns.tolist()
cols = cols[168:] + cols[:168]
X_test = enc_df[cols]


# apply scaler
X_test[features_num_train] = scaler.transform(X_test[features_num_train])


# transform target variable
y_test.replace(-1, 0, inplace=True)


##################################################################################################
#### FEATURE SELECTION
X_train_fs, X_cv_fs, y_train_fs, y_cv_fs = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=3992)

random_classifier = RandomForestClassifier()
parameters = {'max_depth':np.arange(3,10),'n_estimators':list(range(25,251,25))}
random_grid = GridSearchCV(random_classifier, parameters, cv=3)
random_grid.fit(X_cv_fs, y_cv_fs)
print("Optimal Forest Hyperparams:", random_grid.best_params_)

rf_model = RandomForestClassifier(
    n_estimators=25,
    max_depth=3,
    max_features=None,
    max_leaf_nodes=None,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    class_weight='balanced',
    random_state=3992,
    verbose=0,
    warm_start=False)

rf_model.fit(X=X_train_fs, y=y_train_fs)

import pickle
pickle.dump(rf_model, open('data/rf_model.sav', 'wb'))
# rf_model = pickle.load(open('data/rf_model.sav', 'rb'))

feature_importances = pd.DataFrame(rf_model.feature_importances_, index = X_train.columns, columns=['importance']).sort_values('importance', ascending=False)

# feature importance plot
std = np.std([rf_model.feature_importances_[:101] for rf_model in rf_model.estimators_], axis=0)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
feature_importances[:101].plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean Decrease in Impurity (MDI)")
fig.tight_layout()
fig.show()

most_imp_feats = feature_importances[:26].index.to_list()
X_train = X_train[most_imp_feats]
X_test = X_test[most_imp_feats]


##################################################################################################
#### MODEL SELECTION

from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.constraints import maxnorm
from keras.regularizers import l2
from keras.layers import GaussianNoise
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential
import keras.metrics
from sklearn.utils.class_weight import compute_class_weight

def create_model(learning_rate=0.001, dropout_rate=0.0, deep='n', neurons=X_train.shape[1], kernel_initializer='glorot_uniform'):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_dim=X_train.shape[1], kernel_initializer=kernel_initializer))
    model.add(Dropout(dropout_rate))
    if deep == "y":
        model.add(Dense(round(neurons**(1/1.2), 0), activation='relu', input_dim=X_train.shape[1], kernel_initializer=kernel_initializer))
        model.add(Dropout(dropout_rate))
        model.add(Dense(round(neurons**(1/1.5), 0), activation='relu', input_dim=X_train.shape[1], kernel_initializer=kernel_initializer))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name='ROC_AUC'),
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )
    return model

model = KerasClassifier(build_fn=create_model, verbose=2) #this wrapper allows us to feed the Keras model into Sklearn's GridSearchCV class

# defining grid search model; # test architectures first, then tune hyperparamters in optimized architecture
grid_name = "hyparam_grid_"

if grid_name == "arch_grid_":
    param_grid = dict(
        epochs=[5],
        batch_size=[128],
        deep=['n', 'y'],
        neurons=[
            round(X_train.shape[1]**(1/1.5), 0),
            round(X_train.shape[1]/2, 0),
            X_train.shape[1],
            round(X_train.shape[1]*2, 0),
            round(X_train.shape[1]**(1.5), 0),
            round(X_train.shape[1]**(1.6), 0),
            round(X_train.shape[1]**(1.7), 0),
            round(X_train.shape[1]**(1.8), 0),
            round(X_train.shape[1]**(1.9), 0),
            round(X_train.shape[1]**(2), 0),
            round(X_train.shape[1]**(2.1), 0),
            round(X_train.shape[1]**(2.2), 0)]
    )
elif grid_name == "hyparam_grid_":
    param_grid = dict(
        epochs=[5],
        batch_size=[64, 128, 256, 512, 1024],
        deep=['y'],
        neurons=[round(X_train.shape[1]**(2.2), 0)],
        learning_rate=[0.0001, 0.001, 0.01],
        dropout_rate=[0.0, 0.45, 0.9],
        kernel_initializer=['glorot_uniform', 'he_uniform', 'he_normal']
    )
elif grid_name == "TEST_grid_":
    param_grid = dict(
        epochs=[5],
        batch_size=[1024],
        deep=['y'],
        neurons=[round(X_train.shape[1]**(1.9), 0)],
        learning_rate=[0.0001, 0.001, 0.01],
        dropout_rate=[0.0, 0.3, 0.6, 0.9],
        kernel_initializer=['glorot_uniform', 'he_normal', 'he_uniform']
    )

import sklearn.metrics
scoring = {
    "F1": sklearn.metrics.make_scorer(sklearn.metrics.f1_score),
    'ROC_AUC': sklearn.metrics.make_scorer(sklearn.metrics.roc_auc_score),
    "Accuracy": sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score),
    "Recall": sklearn.metrics.make_scorer(sklearn.metrics.recall_score),
    "Precision": sklearn.metrics.make_scorer(sklearn.metrics.precision_score)
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    verbose=3,
    refit='F1',
    n_jobs=2,
    scoring=scoring,
    return_train_score=True,
    cv=StratifiedKFold(n_splits=3, shuffle=True),
)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train.values),
    y=y_train.values.reshape(-1),
)
class_weights = dict(zip(np.unique(y_train.values), class_weights))

grid_result = grid.fit(
    X_train,
    y_train,
    class_weight=class_weights)

# store grid search results
results = pd.DataFrame(grid_result.cv_results_["params"])
results["means_val_F1"] = grid_result.cv_results_["mean_test_F1"]
results['means_val_ROC_AUC'] = grid_result.cv_results_['mean_test_ROC_AUC']
results["means_val_Accuracy"] = grid_result.cv_results_["mean_test_Accuracy"]
results["means_val_Recall"] = grid_result.cv_results_["mean_test_Recall"]
results["means_val_Precision"] = grid_result.cv_results_["mean_test_Precision"]
results["means_train_F1"] = grid_result.cv_results_["mean_train_F1"]
results['means_train_ROC_AUC'] = grid_result.cv_results_['mean_train_ROC_AUC']
results["means_train_Accuracy"] = grid_result.cv_results_["mean_train_Accuracy"]
results["means_train_Recall"] = grid_result.cv_results_["mean_train_Recall"]
results["means_train_Precision"] = grid_result.cv_results_["mean_train_Precision"]

from datetime import datetime
import openpyxl
path = "C:\\Users\\marc.feldmann\\Documents\\data_science_local\\CustomerChurnPrediction\\results\\hyparam_opt\\"
filename = (path + "FNN_clf_GSresults_" + grid_name + datetime.now().strftime("%d_%m_%Y__%H_%M_%S") + ".xlsx")
results.to_excel(filename)


##################################################################################################
#### FINAL MODEL TRAINING

# optimum params:
epochs=5
batch_size=1024
neurons=round(X_train.shape[1]**(1.9), 0)
learning_rate=0.001
dropout_rate=0.0
kernel_initializer='glorot_uniform'

model = Sequential()
model.add(Dense(neurons, activation='relu', input_dim=X_train.shape[1], kernel_initializer=kernel_initializer))
model.add(Dropout(dropout_rate))
model.add(Dense(round(neurons**(1/1.2), 0), activation='relu', input_dim=X_train.shape[1], kernel_initializer=kernel_initializer))
model.add(Dropout(dropout_rate))
model.add(Dense(round(neurons**(1/1.5), 0), activation='relu', input_dim=X_train.shape[1], kernel_initializer=kernel_initializer))
model.add(Dropout(dropout_rate))
model.add(Dense(1, activation='sigmoid'))
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.AUC(name='ROC_AUC'),
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
    ]
)

#### STABLE MODEL
# model = Sequential()
# model.add(Dense(512, activation='relu', input_dim=X_train.shape[1], kernel_initializer='he_normal'))
# model.add(Dropout(0.6))
# model.add(Dense(256, activation='relu', kernel_initializer='he_normal') )
# model.add(Dropout(0.6))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(
#     optimizer='adam',
#     loss="binary_crossentropy",
#     metrics=[
#         keras.metrics.AUC(name='ROC_AUC'),
#         "accuracy",
#         keras.metrics.Precision(name="precision"),
#         keras.metrics.Recall(name="recall")
#     ]
# )

history = model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
    class_weight=class_weights
)

##################################################################################################
#### MODEL TESTING

y_test.shape

y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred, columns=['churn'])
print("ROC-AUC score is {}".format(sklearn.metrics.roc_auc_score(y_test, y_pred)))

precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test, y_pred)
auc = sklearn.metrics.auc(recall, precision)
from matplotlib import pyplot
# no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot(0.0778, 0.0823, marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green", label='Churn Informed Guessing')
# plt.plot([0.4286, 0.4286], [0.375, 0.375], linestyle='--', label='Churn Random Guessing')
plt.plot(recall, precision, marker='.', label='ANN Churn Predictor')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.show()
