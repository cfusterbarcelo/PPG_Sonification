
import pandas as pd
import numpy as np
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

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
plt.show()

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
columns_rf = x.columns_rf[indices_rf]

# Plot mean of feature importance as a lineplot and std as a shaded area
plt.figure(figsize=(10, 7))
plt.plot(columns_rf, importances_rf, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=12)
plt.fill_between(columns_rf, importances_rf - np.std(importances_rf), importances_rf + np.std(importances_rf), alpha=0.2)
plt.xticks(columns_rf, rotation=90)
plt.xlabel('Features')
plt.show()

# TODO: repeat everything for a LG model