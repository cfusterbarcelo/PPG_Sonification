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
| RandomForestClassifier | 0.88 | 0.89 | 0.89 | 0.88 |
| KNeighborsClassifier | 0.88 | 0.89 | 0.89 | 0.88 |


## DL for images approach
Pending. 

## TO DO List
- [ ] Correct Classic AI approach table and finish it for the best 5-6 algorithms.
- [ ] Show best results of Classic AI approach on readme.
- [ ] Run an LGBM model with the features extracted from the audio wave files.
    - [ ] Analyse feature importance
- [ ] Run an XGB model with the features extracted from the audio wave files.
    - [ ] Analyse feature importance
- [ ] Run a Logistic Regressor model with the features extracted from the audio wave files.
    - [ ] Analyse model coeficients.
- [ ] Save all plots in *results* folder.   