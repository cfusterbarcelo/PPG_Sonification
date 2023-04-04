# PPG Sonification

## Introduction
The main objective of this project is to detect weather if a PPG signal has Atrial Fibrillation (AF) or Normal Sinus Rhythm(NSR). 

There are two different approaches in this project:
1. Classic AI.
Each window  from a PPG signal is segmented into 15 seconds as an audio wave file and features are extracted with *insert library*. 

2. DL for images. From the PPG signal, a spectrogram is generated for each 15 seconds window. Then, the spectrogram is converted into an image and the image is fed into a CNN. 

## Classic AI approach
The LazyPredict library is used to train a model with different algorithms. The variable "True" or "False" indicates if the data has been balanced with SMOTE or not. 

| Algorithm | Accuracy | Balanced Accuracy | ROC AUC | F1 Score |
|-----------|----------|-------------------|---------|----------|
| LGBMClassifier | 0.88 | 0.89 | 0.89 | 0.88 | 
| XGBClassifier | 0.88 | 0.89 | 0.89 | 0.88 |
| RandomForestClassifier | 0.87 | 0.87 | 0.87 | 0.87 |
| KNeighborsClassifier | 0.89 | 0.88 | 0.88 | 0.89 |
| LogisticRegression | 0.84 | 0.85 | 0.85 | 0.84 |

### Random Forest Classifier
With a Random Forest (RF) Classifier, the final accuracy achieved is a 90% without balancing with SMOTE. The confusion matrix is the following:

![Confusion Matrix from RF](./results_ppg_sonification_classifiers/cm_knn_False.csv)

### KNearestNeighbors Classifier
The next experiment is done through a KNearestNeighbors (KNN) Classifier. The final accuracy achieved is a 87% without balancing with SMOTE. The confusion matrix is the following:

![Confusion Matrix from KNN](./results_ppg_sonification_classifiers/cm_knn_False.csv)

With a KNN we are able to extract some explainability through analysis of the feature importance. The following plot shows the feature importance of the KNN model:

![Feature importance from KNN](./results_ppg_sonification_classifiers/feature_importance_knn_False.csv)


### LG and LGBM Classifiers

| Algorithm | Accuracy |
|-----------|----------|
| LGBMClassifier | 89.15% |
| GBClassifier | 88.66% |

Confusion matrix from both classifiers are the following:

![Confusion Matrix from LGBM](./results_ppg_sonification_classifiers/cm_lgbm_fb_False.png)

Then, we have been able to also extract feature importance for both models. The following plot shows the feature importance of the LGBM model:

![Feature importance from LGBM](./results_ppg_sonification_classifiers/feature_importance_lgbm_fb_False.png)

THis is the feature importance for the GB model:

![Feature importance from GB](./results_ppg_sonification_classifiers/feature_importance_gb_fb_False.png)


## DL for images approach
Pending. 

## TO DO List
- [X] Correct Classic AI approach table and finish it for the best 5-6 algorithms.
- [X] Show best results of Classic AI approach on readme.
- [X] Run an LGBM model with the features extracted from the audio wave files.
    - [X] Analyse feature importance
- [X] Run an XGB model with the features extracted from the audio wave files.
    - [X] Analyse feature importance
- [ ] Run a Logistic Regressor model with the features extracted from the audio wave files.
    - [ ] Analyse model coeficients.
- [X] Save all plots in *results* folder.  
- [ ] Run a CNN with the results on images. 