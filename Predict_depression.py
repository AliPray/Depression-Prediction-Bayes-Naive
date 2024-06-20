#%%


#import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.stats import poisson
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics


#naive bayes classifier function:
#fit function using gaussian distribution
def fit(X, y):

    # Compute the number of samples and features in the input data.
    n_samples, n_features = X.shape
    
    # Compute the number of unique classes in the target labels.
    classes = np.unique(y)
    n_classes = len(classes)
    
    # Compute the prior probability of each class.
    priors = {}
    for c in classes:
        priors[c] = np.mean(y == c)
        
    # Compute the class conditional probability of each feature for each class.
    conditional_probs = {}
    for c in classes:
        X_c = X[y == c]
        conditional_probs[c] = {}
        for j in range(n_features):
            conditional_probs[c][j] = {}
            unique_feature_values = np.unique(X[:, j])
            for x in unique_feature_values:
                conditional_probs[c][j][x] = np.mean(X_c[:, j] == x)
                
    # Return the prior probabilities and the class conditional probabilities.
    return {"priors": priors, "conditional_probs": conditional_probs}

#prediction function 
def predict(model, X):

    # Compute the number of samples and features in the input data.
    n_samples, n_features = X.shape
    
    # Compute the prior probabilities of each class.
    priors = model["priors"]
    
    # Compute the class conditional probabilities of each feature for each class.
    conditional_probs = model["conditional_probs"]
    
    # Make predictions for each sample in the input data.
    y_pred = np.zeros(n_samples)
    for i in range(n_samples):
        # Initialize the probabilities for each class to the prior probabilities.
        class_probs = {}
        for c in priors:
            class_probs[c] = np.log(priors[c])
            
        # Multiply the probabilities for each feature for each class.
        for j in range(n_features):
            xj = X[i, j]
            for c in class_probs:
                if xj in conditional_probs[c][j]:
                    class_probs[c] += np.log(conditional_probs[c][j][xj] + 1e-10) # add smoothing here
                else:
                    class_probs[c] += np.log(1e-10)
                    
        # Determine the class with the highest probability.
        y_pred[i] = max(class_probs, key=class_probs.get)
        
    # Return the predicted target labels.
    return y_pred



#fit function using poisson distribution
from scipy.stats import poisson

def fitP(X, y):
    # Compute the number of samples and features in the input data.
    n_samples, n_features = X.shape
    
    # Compute the number of unique classes in the target labels.
    classes = np.unique(y)
    n_classes = len(classes)
    
    # Compute the prior probability of each class.
    priors = {}
    for c in classes:
        priors[c] = np.mean(y == c)
        
    # Compute the class conditional probability of each feature for each class.
    conditional_probs = {}
    for c in classes:
        X_c = X[y == c]
        conditional_probs[c] = {}
        for j in range(n_features):
            conditional_probs[c][j] = {}
            unique_feature_values = np.unique(X[:, j])
            for x in unique_feature_values:
                # Use Poisson distribution to compute the probability of a feature value
                # given the class c
                mu = np.mean(X_c[:, j])
                prob = poisson.pmf(x, mu)
                conditional_probs[c][j][x] = prob
                
    # Return the prior probabilities and the class conditional probabilities.
    return {"priors": priors, "conditional_probs": conditional_probs}


#read the excel file to a pandas dataframe
df = pd.read_excel('Freq-PHO-Binary.csv.xlsx', sheet_name='Freq-PHO-Binary.csv', header=0)


#rename the column names:
df.rename(columns={'Gender':'gender','Emotion_Joy': 'joy', 'Emotion_Sadness': 'sadness', 'Emotion_Anger': 'anger', 'Emotion_Disgust': 'disgust', 'Emotion_Fear': 'fear', 'Emotion_Surprise': 'surprise' , 'Emotion_Contempt': 'contempt', 'Emotion_Neutral': 'neutral', 'Depression': 'depression'}, inplace=True)


#change the value of the depression column to normalized value 1 and 0 ( old values: YES AND NO)
df['depression'] = df['depression'].replace('YES', '1')
df['depression'] = df['depression'].replace('NO', '0')
df['gender'] = df['gender'].replace('Male', '1')
df['gender'] = df['gender'].replace('Female', '0')


# calculate class distribution of output column
class_dist = df['depression'].value_counts()

print(class_dist)


#normalize the values of each emotion using mix-max normalization.
#joy
min_valueJoy = df['joy'].min()
max_valueJoy = df['joy'].max()

df['joy'] = df['joy'].apply(lambda x: (x - min_valueJoy) / (max_valueJoy - min_valueJoy))
#sadness
min_valueSad = df['sadness'].min()
max_valueSad = df['sadness'].max()

df['sadness'] = df['sadness'].apply(lambda x: (x - min_valueSad) / (max_valueSad - min_valueSad))
#anger
min_valueAnger = df['anger'].min()
max_valueAnger = df['anger'].max()

df['anger'] = df['anger'].apply(lambda x: (x - min_valueAnger) / (max_valueAnger - min_valueAnger))

#disgust
min_valueDisgust = df['disgust'].min()
max_valueDisgust = df['disgust'].max()

df['disgust'] = df['disgust'].apply(lambda x: (x - min_valueDisgust) / (max_valueDisgust - min_valueDisgust))

#fear
min_valueFear = df['fear'].min()
max_valueFear = df['fear'].max()

df['fear'] = df['fear'].apply(lambda x: (x - min_valueFear) / (max_valueFear - min_valueFear))

#suprise
min_valueSurprise = df['surprise'].min()
max_valueSuprise = df['surprise'].max()

df['surprise'] = df['surprise'].apply(lambda x: (x - min_valueSurprise) / (max_valueSuprise - min_valueSurprise))

#contempt
min_valueContempt = df['contempt'].min()
max_valueContempt = df['contempt'].max()

df['contempt'] = df['contempt'].apply(lambda x: (x - min_valueContempt) / (max_valueContempt - min_valueContempt))

#neutral
min_valueNeutral = df['neutral'].min()
max_valueNeutral = df['neutral'].max()

df['neutral'] = df['neutral'].apply(lambda x: (x - min_valueNeutral) / (max_valueNeutral - min_valueNeutral))

#print the dataframe
print(df)


#before trainign visualization
#histograms
df.hist(bins=10, figsize=(10,10))
plt.show()

# Split the dataset into input variables (X) and output variable (y)
inputs = df.drop('depression', axis=1)
outputs = df['depression']

# Create a scatter plot matrix
print("scatter plot matrix per pair:")
sns.pairplot(df, x_vars=inputs.columns, y_vars=['depression'])
plt.show()

# Create a box plot
print("box plot:")
sns.boxplot(data=inputs, palette='Set3')
plt.show()

# Create a violin plot
print("violin plot:")
sns.violinplot(data=inputs, palette='Set3')
plt.show()


# cast all columns to float
df = df.astype(float)

#pre processing: create 2 dframes for inputs and outputs 
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#train and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Naive Bayes classifier on training data
model = fit(X_train, y_train)
modelP= fitP(X_train, y_train)

y_pred = predict(model, X_test)
y_predP = predict(modelP, X_test)
# Evaluate predictions
accuracy = np.mean(y_pred == y_test)
accuracyP = np.mean(y_predP == y_test)

# Compute the recall, F1 score, and ROC curve

print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 score:", metrics.f1_score(y_test, y_pred))
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)

print("ROC AUC:", roc_auc)

#confusion matrix 
print("confusion matrix:")
# Assuming you have trained your model and have a `y_pred` vector of predicted labels
conf_mat = confusion_matrix(y_test, y_predP)
# Create a heatmap of the confusion matrix using seaborn
sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Fit a ZeroR classifier and predict the test set
zero_r = DummyClassifier(strategy="most_frequent")
zero_r.fit(X_train, y_train)
zero_r_pred = zero_r.predict(X_test)

# Fit a OneR classifier and predict the test set
one_r = KNeighborsClassifier(n_neighbors=1)
one_r.fit(X_train, y_train)
one_r_pred = one_r.predict(X_test)

#naive bayes accuracy
print("Accuracy Naivebayes(using Gaussian distribution):", accuracy)
print("Accuracy Naivebayes(using Poisson distribution):", accuracyP)

# Calculate and print the accuracy of each classifier; zeroR and oneR
print('zeroR and oneR performance:')
print("ZeroR Accuracy:", accuracy_score(y_test, zero_r_pred))
print("OneR Accuracy:", accuracy_score(y_test, one_r_pred))


k = 5 # number of folds
kf = KFold(n_splits=k)

accuracies = []
for train_index, test_index in kf.split(X):
    # Split the data into train and test sets for this fold
    X_train2, X_test2 = X[train_index], X[test_index]
    y_train2, y_test2 = y[train_index], y[test_index]

    # Train the model using the fit function
    model2 = fitP(X_train2, y_train2)

    # Use the predict function to make predictions on the test set
    y_pred = predict(model2, X_test2)

    # Compute the accuracy of the predictions for this fold
    accuracy = np.mean(y_pred == y_test2)
    accuracies.append(accuracy)

# Compute the mean and standard deviation of the accuracies across all folds
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

print("mean accuracy of 10 k folds: ",mean_accuracy)
print("standard deviation accuracy of 5 k folds: ",std_accuracy)



# %%
