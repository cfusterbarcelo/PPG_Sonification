
import pandas as pd
import numpy as np
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from lazypredict.Supervised import LazyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb



test_size = 0.2
balance = False

# ============================================
# =============== READ DATASET ===============
# ============================================
# Read csv data: 
df = pd.read_csv('./data/salida15sec.arff.csv', sep=',')
# Split data and labels
## x and y: x = data, y = last column
x = df.iloc[:, :-1]
y = df.iloc[:, -1]


# ============================================
# =================== EDA ====================
# ============================================
## Convert y into numbers class1 = 0, class2 = 1... to treat it like a binary class
y = pd.factorize(y)[0]

## Check if there are missing values
missing = x.isnull().sum().sum()
if missing: print('Missing values: ', missing)

## Split test and train data (before standarise) using sklearn
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=31)

## Check data inbalance (y_train being numpy array)
### Note: counts = [1024, 1216]
unique, counts = np.unique(y_train, return_counts=True)
## Correct inbalance using SMOTE only if balance = True
if balance:
    sm = SMOTE(random_state=31)
    x_train, y_train = sm.fit_resample(x_train, y_train)


## Standarise data using standard scaler
scaler = StandardScaler()
### Learn the mean and variance of each column of the data
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# ============================================
# ==================== MODELS ================
## Use lazypredict to find the best model (no crossvalidation)
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)
print(models)
# Export results to csv
models.to_csv('./results/results_'+str(balance)+'.csv')

## ============ KNN MODEL AND RESULTS =========
## MODEL CROSSVALIDATION
## Crossvalidate KNeighborsClassifier (best model) with gridsearch
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 25)}
knn_gscv = GridSearchCV(knn, param_grid, cv=5)
knn_gscv.fit(x_train, y_train)
select_best_model = knn_gscv.best_estimator_

## RESULTS
prediction_knn = select_best_model.predict(x_test)
cm_knn = confusion_matrix(y_test, prediction_knn)
# Store cm_knn in a file
cm_knn = pd.DataFrame(cm_knn)
cm_knn.to_csv('./results/cm_knn_'+str(balance)+'.csv')
print(cm_knn)
cr_knn = classification_report(y_test, prediction_knn)
print(cr_knn)
### Store results in a dictionary
results_knn = {
    'model': select_best_model,
    'confusion_matrix': cm_knn,
    'classification_report': cr_knn
}
### Export results to pickle
pickle.dump(results_knn, open('./results/results_knn_'+str(balance)+'.pkl', 'wb'))

# ================== PLOTS for KNN ===================
## Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm_rf, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix for KNN')
plt.show()

## Extract feature importance that are only > 0
importances_knn = best_model.feature_importances_
indices_knn = np.where(importances_knn > 0)[0]
importances_knn = importances_knn[indices_knn]
columns_knn = x.columns[indices_knn]

# Plot mean of feature importance as a lineplot and std as a shaded area
plt.figure(figsize=(10, 7))
plt.plot(columns_knn, importances_knn, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=12)
plt.fill_between(columns_knn, importances_knn - np.std(importances_knn), importances_knn + np.std(importances_knn), alpha=0.2)
plt.xticks(columns_knn, rotation=90)
plt.xlabel('Features')
plt.title('KNN Feature Importance')
plt.show()
plt.savefig('./results/feature_importance_knn_'+str(balance)+'.png')


## ================== RF MODEL AND RESULTS ===================
## MODEL CROSSVALIDATION
## Crossvalidate RandomForestClassifier (best model) with gridsearch
rf = RandomForestClassifier()
print("Cross-validating using grid search...")
grid = GridSearchCV(
    estimator=rf,
    param_grid={
        "n_estimators": [100],
        "max_depth": [2, 4, 6, 8],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt", "log2"],
    },
    scoring="balanced_accuracy",
    cv=5,
    verbose=2,
)
grid_results = grid.fit(x_train, y_train)
best_model = grid_results.best_estimator_

## RESULTS
prediction_rf = select_best_model.predict(x_test)
cm_rf = confusion_matrix(y_test, prediction_rf)
# Store cm_knn in png file
cm_rf = pd.DataFrame(cm_rf)
cm_rf.to_csv('./results/cm_rf_'+str(balance)+'.csv')
print(cm_rf)
cr_rf = classification_report(y_test, prediction_rf)
print(cr_rf)
### Store results in a dictionary
results_rf = {
    'model': select_best_model,
    'confusion_matrix': cm_rf,
    'classification_report': cr_rf
}
### Export results to pickle
pickle.dump(results_rf, open('./results/results_rf_'+str(balance)+'.pkl', 'wb'))

# ================== PLOTS for RF ===================
# TODO: repeat for knn
## Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm_rf, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix for RF')
plt.show()

## Extract feature importance that are only > 0
importances_rf = best_model.feature_importances_
indices_rf = np.where(importances_rf > 0)[0]
importances_rf = importances_rf[indices_rf]
columns_rf = x.columns[indices_rf]

# Plot mean of feature importance as a lineplot and std as a shaded area
plt.figure(figsize=(10, 7))
plt.plot(columns_rf, importances_rf, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=12)
plt.fill_between(columns_rf, importances_rf - np.std(importances_rf), importances_rf + np.std(importances_rf), alpha=0.2)
plt.xticks(columns_rf, rotation=90)
plt.xlabel('Features')
plt.title('Random Forest Feature Importance')
plt.show()
plt.savefig('./results/feature_importance_rf_'+str(balance)+'.png')

## ================== LGBM MODEL AND RESULTS ===================
# Crossvalidate LGBMClassifier (best model) with gridsearch
lgbm_params = {
    'boosting_type': ['gbdt', 'dart', 'goss'],
    'num_leaves': [10, 20, 30],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [50, 100, 200],
    'reg_alpha': [0, 1, 5],
    'reg_lambda': [0, 1, 5],
    'random_state': [42]
}
gb_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1],
    'random_state': [42]
}
lgbm_clf = lgb.LGBMClassifier()
gb_clf = GradientBoostingClassifier()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_lgbm = GridSearchCV(lgbm_clf, param_grid=lgbm_params, cv=kf, scoring='accuracy')
grid_lgbm.fit(x_train, y_train)

grid_gb = GridSearchCV(gb_clf, param_grid=gb_params, cv=kf, scoring='accuracy')
grid_gb.fit(x_train, y_train)
grid_model = grid_lgbm.best_estimator_

print("Best parameters for LGBM:", grid_lgbm.best_params_)
print("Accuracy score for LGBM:", grid_lgbm.best_score_)

print("Best parameters for GB:", grid_gb.best_params_)
print("Accuracy score for GB:", grid_gb.best_score_)

## RESULTS
y_pred_lgbm = grid_lgbm.predict(x_test)
y_pred_gb = grid_gb.predict(x_test)

cm_lgbm = confusion_matrix(y_test, y_pred_lgbm)
cm_gb = confusion_matrix(y_test, y_pred_gb)

report_lgbm = classification_report(y_test, y_pred_lgbm)
report_gb = classification_report(y_test, y_pred_gb)

print(report_lgbm)
print(report_gb)
### Store results in a dictionary
results = {}
results['LGBM Classifier'] = {
    'model': grid_lgbm.best_estimator_,
    'confusion_matrix': cm_lgbm,
    'classification_report': report_lgbm
}

results['Gradient Boosting Classifier'] = {
    'model': grid_gb.best_estimator_,
    'confusion_matrix': cm_gb,
    'classification_report': report_gb
}
### Export results to pickle
pickle.dump(results, open('./results/results_lgbm_'+str(balance)+'.pkl', 'wb'))

# ================== PLOTS for LGBM ===================
# CM
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.heatmap(cm_lgbm, annot=True, cmap='Blues')
plt.title('Confusion Matrix for LGBM Classifier')
plt.subplot(1,2,2)
sns.heatmap(cm_gb, annot=True, cmap='Blues')
plt.title('Confusion Matrix for Gradient Boosting Classifier')
plt.savefig('./results/cm_lgbm_gb_'+str(balance)+'.png')
plt.show()

## Extract feature importance that are only > 0 for LGBM
importances_lgbm = grid_model.feature_importances_
indices_lgbm = np.where(importances_lgbm > 0)[0]
importances_lgbm = importances_lgbm[indices_lgbm]
columns_lgbm = x.columns[indices_lgbm]

## Extract feature importance that are only < 0 for GB
importances_gb = grid_gb.best_estimator_.feature_importances_
indices_gb = np.where(importances_gb > 0)[0]
importances_gb = importances_gb[indices_gb]
columns_gb = x.columns[indices_gb]

# Plot mean of feature importance as a lineplot and std as a shaded area
plt.figure(figsize=(10, 7))
plt.plot(columns_lgbm, importances_lgbm, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=12)
plt.fill_between(columns_lgbm, importances_lgbm - np.std(importances_lgbm), importances_lgbm + np.std(importances_lgbm), alpha=0.2)
plt.xticks(columns_lgbm, rotation=90)
plt.xlabel('Features')
plt.title('LGBM Feature Importance')
plt.show()
plt.savefig('./results/feature_importance_lgbm_'+str(balance)+'.png')

plt.figure(figsize=(10, 7))
plt.plot(columns_gb, importances_gb, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=12)
plt.fill_between(columns_gb, importances_gb - np.std(importances_gb), importances_gb + np.std(importances_gb), alpha=0.2)
plt.xticks(columns_gb, rotation=90)
plt.xlabel('Features')
plt.title('GB Feature Importance')
plt.show()
plt.savefig('./results/feature_importance_gb_'+str(balance)+'.png')
