import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from numpy.random import seed
import tensorflow as tf
seed(3992)
tf.random.set_seed(3992)

import pandas as pd
pd.set_option("float_format", "{:f}".format)
pd.set_option('display.max_columns', None)
X = pd.read_table('data/orange_small_train.data')
y = pd.read_table('data/orange_small_train_churn.labels', header=None, names=['Churn'])
# data = pd.concat([X, y], axis=1)
# X.head()
# X.info()

# # change feature data types to float (nums) or object (cats)
# X_clm_dtypes = {clm: X[clm].dtype for clm in X.columns}
# for clm in X.columns:
#     if X_clm_dtypes[clm] == int:
#         x = X[clm].astype(float)
#         X.loc[:, clm] = x
#         X_clm_dtypes[clm] = x.dtype
#     elif X_clm_dtypes[clm] != float:
#         x = X[clm].astype('object')
#         X.loc[:, clm] = x
#         X_clm_dtypes[clm] = x.dtype

# round(X.isna().sum().sum()/(X.shape[0]*X.shape[1]), 3)

# import matplotlib.pyplot as plt
# temp = X.isna().sum()/(X.shape[0])
# plt.bar(range(len(temp)), sorted(temp), color='blue', alpha=0.65)

# plt.hist(y['Churn'], bins=3)
# y['Churn'].value_counts()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=3992)

X_train.info()
X_test.info()

X_train.dropna(axis=0, how='all', inplace=True)
X_train.dropna(axis=1, how='all', inplace=True)

# temp = X_train.isna().sum()/(X_train.shape[0])
# plt.bar(range(len(temp)), sorted(temp), color='blue', alpha=0.65)

# X_train.shape

X_train['Var73'] = X_train['Var73'].astype('float64')

# turn all categorical features into data type 'category'
# (how to find the proper cutoff calue?)
features_cat_train = list(X_train.select_dtypes(include=['object']).columns)
for clm in features_cat_train:
    X_train.loc[X_train[clm].value_counts(dropna=False)[X_train[clm]].values < X_train.shape[0] * 0.015, clm] = "RARE_VALUE"


##################
##################

for iteration, clm in enumerate(features_cat_train):
    print(
        "Encoding categorical variable (training set)",
        iteration + 1,
        "/ ",
        len(features_cat_train))
    most_freq_vals = X_train[clm].value_counts()[:20].index.tolist()
    dummy_clms = pd.get_dummies(X_train[clm].loc[X_train[clm].isin(most_freq_vals)], prefix=clm)
    X_train = pd.merge(
        X_train,
        dummy_clms,
        left_index=True,
        right_index=True,
        how='outer')
    for dum_clm in X_train[dummy_clms.columns]:
        X_train[dum_clm].fillna(0, inplace=True)
    X_train.drop(clm, axis=1, inplace=True)

##################
##################

X_train.info()

# X_clm_dtypes = {clm: X_train[clm].dtype for clm in X_train.columns}
# for clm in X_train.columns:
#     if X_clm_dtypes[clm] == int:
#         x = X_train[clm].astype(float)
#         X_train.loc[:, clm] = x
#         X_clm_dtypes[clm] = x.dtype
#     elif X_clm_dtypes[clm] != float:
#         x = X_train[clm].astype('object')
#         X_train.loc[:, clm] = x
#         X_clm_dtypes[clm] = x.dtype

# insert NaN indicator columns
import numpy as np
for clm in X_train:
    if X_train[clm].isna().sum() > 0:
        X_train.insert(X_train.shape[1], f"{clm}_NaNInd", 0)
        X_train[f"{clm}_NaNInd"] = np.where(np.isnan(X_train[clm]), 1, 0)

X_train.info()

# X_clm_dtypes = {clm: X_train[clm].dtype for clm in X_train.columns}
# for clm in X_train.columns:
#     if X_clm_dtypes[clm] == int:
#         x = X_train[clm].astype(float)
#         X_train.loc[:, clm] = x
#         X_clm_dtypes[clm] = x.dtype
#     elif X_clm_dtypes[clm] != float:
#         x = X_train[clm].astype('object')
#         X_train.loc[:, clm] = x
#         X_clm_dtypes[clm] = x.dtype

X_train.fillna(X.median(), inplace=True)

round(X_train.isna().sum().sum()/(X_train.shape[0]*X.shape[1]), 3)

y_train['Churn'] = (y['Churn'] + 1) / 2
y_train.value_counts()


######### prep test set
X_test.dropna(0, how='all', inplace=True)
X_test.dropna(1, how='all', inplace=True) 

X_test['Var73'] = X_test['Var73'].astype('float64')
features_cat_test = list(X_test.select_dtypes(include=['object']).columns)
for clm in features_cat_test:
    X_test.loc[X_test[clm].value_counts(dropna=False)[X_test[clm]].values < X_test.shape[0] * 0.015, clm] = "RARE_VALUE"

for iteration, clm in enumerate(features_cat_test):
    print(
        "Encoding categorical variable (test set)",
        iteration + 1,
        "/ ",
        len(features_cat_test))
    most_freq_vals = X_test[clm].value_counts()[:20].index.tolist()
    dummy_clms = pd.get_dummies(X_test[clm].loc[X_test[clm].isin(most_freq_vals)], prefix=clm)
    X_test = pd.merge(
        X_test,
        dummy_clms,
        left_index=True,
        right_index=True,
        how='outer')
    for dum_clm in X_test[dummy_clms.columns]:
        X_test[dum_clm].fillna(0, inplace=True)
    X_test.drop(clm, axis=1, inplace=True)

for clm in X_test:
    if X_test[clm].isna().sum() > 0:
        X_test.insert(X_test.shape[1], f"{clm}_NaNInd", 0)
        X_test[f"{clm}_NaNInd"] = np.where(np.isnan(X_test[clm]), 1, 0)

X_test.fillna(X.median(), inplace=True)

y_test['Churn'] = (y['Churn'] + 1) / 2


## ensure that test data has same features in same order as those the model has learned from the train data
### features that are in train and test data > retain in test data
### features that are in train data only > drop
features_train_only = X_train.columns.difference(X_test.columns) 
X_train.drop(labels=features_train_only, axis=1, inplace=True)

### features that are in test data only > drop
features_test_only = X_test.columns.difference(X_train.columns) 
X_test.drop(labels=features_test_only, axis=1, inplace=True)

### ensure features in test data follow feature order in train data
X_test = X_test[X_train.columns]

### test whether train and test data columns are identical (features and feature order)
# X_test.columns == X_train.columns

X_test.isna().sum().sum()

# features_f64_train = list(X_train.select_dtypes(include=['float64']).columns)
# features_f64_test = list(X_test.select_dtypes(include=['float64']).columns)
# X_train[features_f64_train] = X_train[features_f64_train].astype('float32')
# X_train = np.asarray(X_train).astype('float32')

# preserve preprocessesed and feature-harmonized, still unscaled train and test sets
X_train_pp = X_train.copy()
X_test_pp = X_test.copy()

# select numerical columns that are not binary

features_num_train_nonbinary = X_train.iloc[:, :174].columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[features_num_train_nonbinary] = scaler.fit_transform(X_train[features_num_train_nonbinary])


#### FEATURE SELECTION
# create random subsets of training data for optimizing the feature selection mode to reduce computation time
_, X_train_fs, _, y_train_fs = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=3992)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
random_classifier = RandomForestClassifier()
parameters = { 'max_depth':np.arange(5,10),'n_estimators':list(range(75,301,25))}
random_grid = GridSearchCV(random_classifier, parameters)
random_grid.fit(X_train_fs, np.array(y_train_fs['Churn']))
print("Optimal Hyperarams: ", random_grid.best_params_)

rf_model = RandomForestClassifier(
    n_estimators=75,
    max_depth=5,
    max_features=None,   
    max_leaf_nodes=None,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    class_weight='balanced',
    random_state=3992,
    verbose=0,
    warm_start=False)

# search features on full training set
rf_model.fit(X_train, np.array(y_train['Churn']))

import pickle
pickle.dump(rf_model, open('data/rf_model.sav', 'wb'))
rf_model = pickle.load(open('data/rf_model.sav', 'rb'))

feature_importances = pd.DataFrame(rf_model.feature_importances_, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)

# feature importance plot
# std = np.std([rf_model.feature_importances_[:101] for rf_model in rf_model.estimators_], axis=0)
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# feature_importances[:101].plot.bar(yerr=std, ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean Decrease in Impurity (MDI)")
# fig.tight_layout()
# fig.show()

# reduce preserved training and test datasets to most important features
most_imp_feat = feature_importances[:100].index.to_list()
X_train = X_train_pp[most_imp_feat]
X_test = X_test_pp[most_imp_feat]

# scale
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

X_train.info()
X_test.info()

##################
##################
output_dim = 1
input_dim = X_train.shape[1]

batch_size = 512
nb_epoch = 50

from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.constraints import maxnorm
from keras.regularizers import l2
from keras.layers import GaussianNoise
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential
import keras.metrics

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(input_dim,), kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu', kernel_initializer='he_normal') )
model.add(Dropout(0.8))
model.add(Dense(output_dim, activation='sigmoid'))
model.summary()

##################
##################

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=8, verbose=1, validation_split=0.2)


from sklearn import metrics
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred, columns = ['churn'])
print("ROC-AUC score is {}".format(metrics.roc_auc_score(y_test, y_pred)))

import matplotlib.pyplot as plt
precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)
auc = metrics.auc(recall, precision)
no_skill = y_test.sum() / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Churn Informed Guessing')
plt.plot(recall, precision, marker='.', label='ANN Churn Predictor')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()