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


##### SPLIT DATA
X = pd.read_table('data/orange_small_train.data')
y = pd.read_table('data/orange_small_train_churn.labels', header = None,sep='\t').loc[:, 0].astype('category')

y.value_counts().plot.bar()
# plt.ylabel('value')
# plt.title('churn value for each class')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


##################################################################################################
##### FIT AND APPLY DATA PREP ON TRAINING SET


# imputation, encoding, scaling

### 1) BRING INTO RIGHT ORDER, 2) adapt code

# change datatypes
train['Var73']=train['Var73'].astype('float')

# remove all NaN rows and columns, store these rows and columns for later application to test data
train.dropna(axis=0, how='all', inplace=True)
train.dropna(axis=1, how='all', inplace=True)

# fit and apply missing value imputer
>>> from sklearn.impute import SimpleImputer
>>> imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
>>> imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])

train.fillna(train.median(), inplace=True)

# Create NaN indicator columns and store these columns for test data preprocessing
for clm in train:
    if train[clm].isna().sum() > 0:
        train.insert(train.shape[1], f"{clm}_NaNInd", 0)
        train[f"{clm}_NaNInd"] = np.where(pd.isnull(train[clm]), 1, 0)

# Replace all infrequent cat values with same value, store list of all replaced values
features_cat = list(train.select_dtypes(include=['object']).columns)
for clm in features_cat:
    train.loc[train[clm].value_counts(dropna=False)[train[clm]].values < train.shape[0] * 0.015, clm] = 'RARE_VALUE'
train[train == 'RARE_VALUE'].count().sum()

# Encode categorical features
train_data_1 = train.copy()
for i, clm in enumerate(features_cat):
    print("Encoding categorical variable", i+1, "/ ", len(features_cat))
    most_freq_vals = train[clm].value_counts()[:20].index.tolist()
    dummy_clms = pd.get_dummies(train[clm].loc[train[clm].isin(most_freq_vals)], prefix=clm)
    train_data_1 = pd.merge(
        train_data_1,
        dummy_clms,
        left_index=True,
        right_index=True,
        how='outer')
    for dum_clm in train_data_1[dummy_clms.columns]:
        train_data_1[dum_clm].fillna(0, inplace=True)
    train_data_1.drop(clm, axis=1, inplace=True)

# fit and apply scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# transform target variable
y_train.replace(-1, 0, inplace=True)
y_test.replace(-1, 0, inplace=True)


##################################################################################################
##### APPLY DATA PREP TEST SET


##################################################################################################
#### EVALUATE MODEL






##################################################################################################
##### CHECKED: THESE CODE BITS CAN INDIVIDUALLY CONSIDERED NOT BE THE SCRIPTS 'SECRET SAUCE' #####
# for i in train.columns:
#     if train[i].isnull().sum()/len(train)>=1.0:
#         train.drop(i,axis=1,inplace=True)
##################################################################################################


##################################################################################################
##### CHECKED: THESE CODE BITS CAN INDIVIDUALLY CONSIDERED NOT BE THE SCRIPTS 'SECRET SAUCE' #####
# features_num = list(train.select_dtypes(exclude=['object']).columns)
# float_x_means = train.mean()
# for feat in features_num:
#     x = train[feat]
#     mediancol=train[feat].mean()
#     isThereMissing = x.isnull()
#     if isThereMissing.sum() > 0:
#         train.loc[isThereMissing.tolist(), feat] = float_x_means[feat]
##################################################################################################


##################################################################################################
##### CHECKED: THESE CODE BITS CAN INDIVIDUALLY CONSIDERED NOT BE THE SCRIPTS 'SECRET SAUCE' #####
# categorical_levels = train[categorical_DataVars].apply(lambda col: len(col.cat.categories))
# categorical_x_var_names = categorical_levels[categorical_levels > 10].index
##################################################################################################


##################################################################################################
##### CHECKED: THESE CODE BITS CAN INDIVIDUALLY CONSIDERED NOT BE THE SCRIPTS 'SECRET SAUCE' #####
# the following function replaces all categories in all cat variables with more than 10 categories by 'OTHER'
# this significantly reduces variation in the dataset - makes it easier for ANN to learn?
# def replaceInfrequentLevels(data, val=0.015):
#     collapsed_categories = {}
#     for categorical_x_var_name in categorical_x_var_names:
#         x = data[categorical_x_var_name].copy()
#         for category in x.cat.categories:
#             matching_rows_yesno = x == category
#             if matching_rows_yesno.sum() < val * data.shape[0]: #wenn kategorie weniger als 1.5% der reihen belegt
#                 if categorical_x_var_name in collapsed_categories:
#                     collapsed_categories[categorical_x_var_name].append(category)
#                 else:
#                     collapsed_categories[categorical_x_var_name] = [category]
#                 if 'OTHER' not in data[categorical_x_var_name].cat.categories:
#                     data[categorical_x_var_name].cat.add_categories('OTHER', inplace=True)
#                 data.loc[matching_rows_yesno, categorical_x_var_name] = 'OTHER'
#                 data[categorical_x_var_name].cat.remove_categories(category, inplace=True)
#     return data

# train = replaceInfrequentLevels(train)
##################################################################################################


##################################################################################################
##### CHECKED: THESE CODE BITS CAN INDIVIDUALLY CONSIDERED NOT BE THE SCRIPTS 'SECRET SAUCE' #####
# categorical_x_nb_levels = train[categorical_x_var_names].apply(lambda col: len(col.cat.categories))
# print("Number of unique values within categorical features after preprocessing:")
# categorical_x_nb_levels

# for categorical_var_name in categorical_x_var_names:
#     train[categorical_var_name].cat.add_categories("unknown_"+categorical_var_name, inplace=True)
#     train[categorical_var_name].fillna("unknown_"+categorical_var_name, inplace=True)
##################################################################################################


##################################################################################################
##### CHECKED: THESE CODE BITS CAN INDIVIDUALLY CONSIDERED NOT BE THE SCRIPTS 'SECRET SAUCE' #####
# train_data_1 = pd.get_dummies(train, columns=categorical_DataVars)
##################################################################################################


##################################################################################################
##### CHECKED: THESE CODE BITS CAN INDIVIDUALLY CONSIDERED NOT BE THE SCRIPTS 'SECRET SAUCE' #####
# def standardize(train, test):
#     mean = np.mean(train, axis=0)
#     std = np.std(train, axis=0)+0.000001

#     X_train = (train - mean) / std
#     X_test = (test - mean) /std
#     return X_train, X_test

# X_train, X_test=standardize(X_train, X_test)
##################################################################################################


### FEATURE SELECTION
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=42)

random_classifier = RandomForestClassifier()
parameters = { 'max_depth':np.arange(5,10),'n_estimators':list(range(75,301,25))}
random_grid = GridSearchCV(random_classifier, parameters, cv = 3)
random_grid.fit(X_cv, y_cv)
print("Best HyperParameter: ",random_grid.best_params_)

rf_model = RandomForestClassifier(
    n_estimators=75,
    max_depth=5,
    max_features=None,
    max_leaf_nodes=None,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    class_weight='balanced',
    random_state=42,
    verbose=0,
    warm_start=False)

rf_model.fit(X=X_train, y=y_train)

feature_importances = pd.DataFrame(rf_model.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
most_imp_feat = feature_importances[:51].index.to_list()

train_data_1=train_data_1[most_imp_feat]


# feature importance plot
# std = np.std([rf_model.feature_importances_[:101] for rf_model in rf_model.estimators_], axis=0)
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# feature_importances[:101].plot.bar(yerr=std, ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean Decrease in Impurity (MDI)")
# fig.tight_layout()
# fig.show()


##################################################################################################
##### CHECKED: THESE CODE BITS CAN INDIVIDUALLY CONSIDERED NOT BE THE SCRIPTS 'SECRET SAUCE' #####
# lst1=[]
# for i in train_data_1.columns:
#     if train_data_1[i].dtypes==float:
#         lst1.append(i)

# from sklearn.preprocessing import StandardScaler
# def standardize_data(X_train, X_test):
#     float_feats_train = X_train[lst1]
#     float_feats_test = X_test[lst1]
#     scaler = StandardScaler()
#     X_train_stand = pd.DataFrame(scaler.fit_transform(float_feats_train), columns = lst1, index = X_train.index)
#     X_test_stand = pd.DataFrame(scaler.transform(float_feats_test), columns = lst1, index = X_test.index)
#     X_train_stand = pd.merge(X_train_stand, X_train.loc[:, ~X_train.columns.isin(lst1)], left_index=True, right_index=True)
#     X_test_stand = pd.merge(X_test_stand, X_test.loc[:, ~X_test.columns.isin(lst1)], left_index=True, right_index=True)
#     return X_train_stand, X_test_stand
##################################################################################################


X_train, X_test, y_train, y_test = train_test_split(train_data_1, churn, stratify=churn, test_size=0.2, random_state=42)
temp = train_data_1.select_dtypes(include=['float']).columns.to_list()
X_train[temp] = scaler.fit_transform(X_train[temp])
X_test[temp] = scaler.transform(X_test[temp])


##################################################################################################
##### CHECKED: THESE CODE BITS CAN INDIVIDUALLY CONSIDERED NOT BE THE SCRIPTS 'SECRET SAUCE' #####
# X_train, X_test=standardize_data(X_train, X_test)
##################################################################################################


y_train.replace(-1, 0, inplace=True)
y_test.replace(-1, 0, inplace=True)


output_dim = 1
input_dim = X_train.shape[1]

batch_size = 512
nb_epoch = 50

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(input_dim,), kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu', kernel_initializer='he_normal') )
model.add(Dropout(0.8))
model.add(Dense(output_dim, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train.values, np.array(y_train.values), batch_size=batch_size, epochs=8, verbose=1, validation_split=0.2)

y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred, columns = ['churn'])
print("ROC-AUC score is {}".format(metrics.roc_auc_score(y_test, y_pred)))

precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)
auc = metrics.auc(recall, precision)
from matplotlib import pyplot
no_skill = len(y_test[y_test==1]) / len(y_test)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Churn Random Guessing')
pyplot.plot(recall, precision, marker='.', label='ANN Churn Predictor')
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.legend()
pyplot.show()