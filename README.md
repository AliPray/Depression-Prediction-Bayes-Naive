# Depression Prediction Model

## Overview
This project involves developing a Naive Bayes model from scratch to predict the presence of depression based on emotion inputs. The model uses the following emotional states as features: joy, sadness, anger, disgust, fear, surprise, contempt, and neutral. The goal is to evaluate the performance of this model using various metrics and to compare it against ZeroR and OneR classifiers.

## Features
- **Emotions**: 
  - Joy
  - Sadness
  - Anger
  - Disgust
  - Fear
  - Surprise
  - Contempt
  - Neutral

## Model Description
### Naive Bayes Classifier
The Naive Bayes classifier is built from scratch using two different distribution algorithms:
1. **Gaussian Naive Bayes**: Assumes that the features follow a Gaussian (normal) distribution.
2. **Poisson Naive Bayes**: Assumes that the features follow a Poisson distribution.

### Comparison Models
- **ZeroR Classifier**: A baseline model that predicts the majority class for all instances.
- **OneR Classifier**: A simple model that creates one rule for each predictor and selects the rule with the least error.

## Evaluation Metrics
The performance of the models is evaluated using the following metrics:
- **Accuracy**: The proportion of correctly classified instances.
- **F1 Score**: The harmonic mean of precision and recall.
- **Recall**: The proportion of actual positives that are correctly identified.
- **ROC AUC**: The area under the receiver operating characteristic curve.
- **Confusion Matrix**: A table used to describe the performance of the classification model.
- **Mean Accuracy**: The average accuracy obtained from k-fold cross-validation.
- **Standard Deviation**: The standard deviation of the accuracy across k-folds.

## K-Fold Cross-Validation
The models are evaluated using k-fold cross-validation with different values of k. This technique helps in assessing the robustness and generalization ability of the models.

## Comparison and Results
The performance of the Naive Bayes models (Gaussian and Poisson) is compared with the ZeroR and OneR classifiers. The comparison includes the metrics mentioned above to determine the best model for predicting depression based on emotional inputs.

## Conclusion
This project provides a comprehensive analysis of a Naive Bayes model's ability to predict depression using emotion-based features. By comparing it with simple baseline models like ZeroR and OneR, we gain insights into its effectiveness and practical utility.

## How to Use
1. **Data Preparation**: Ensure your dataset is formatted with the appropriate emotional features.
2. **Model Training**: Train the Naive Bayes model using either Gaussian or Poisson distribution algorithms.
3. **Evaluation**: Use the provided evaluation metrics to assess the model's performance.
4. **Comparison**: Compare the Naive Bayes model's performance against ZeroR and OneR classifiers.

## Dependencies
- Python 3.x
- NumPy
- Pandas
- Scikit-learn (for evaluation metrics)
- Matplotlib (for visualizing ROC curves and confusion matrices)
