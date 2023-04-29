from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
import warnings # current version generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
# Project Part 2 - Simulations for Spam and Phishing Detection

# Finding accuracy score of Naive Bayes on Titanic Data & correlation between some columns
raw_data = pd.read_csv('customized_dataset_small.csv')  # Reads in data

data = raw_data.dropna(axis=0)  # Drop rows with missing data

df = data.drop('phishing', axis=1)  # Separates the data we're analyzing
label = data['phishing']  # Saves column for doing spilt to find accuracy

scaler = StandardScaler()  # Apply scaling to the dataset
scaler.fit(df)  # Fit scale
X_scaled = scaler.transform(df)  # Transform scale

pca2 = PCA(n_components=2)  # Apply PCA to reduce dimensionality
principalComponents = pca2.fit_transform(X_scaled)  # Fit and transformed scaled dataset

principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])  # Move PCA results into array

finalDf = pd.concat([principalDf, data[['phishing']]], axis=1)  # Concat results with target

X = finalDf.drop('phishing', axis=1)  # Separates the data we're analyzing
Y = finalDf['phishing']  # Saves column for doing spilt to find accuracy

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)  # Spilt training data

classifier = BernoulliNB()  # Create instance of BernoulliNB class
classifier.fit(X_train, y_train)  # Fit data to model

y_pred = classifier.predict(X_test)  # Predict data from model

print("Classification Report: \n", classification_report(y_test, y_pred))
print('Accuracy is', accuracy_score(y_pred, y_test))  # Accuracy score
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))  # Print Confusion matrix

print("\nSVM Results: ")
classifier_svc = SVC()  # Do same as above with SVC
classifier_svc.fit(X_train, y_train)
y_pred = classifier_svc.predict(X_test)

# Summary of the predictions made by the classifier
print("Classification Report: \n", classification_report(y_test, y_pred))
print('Accuracy is', accuracy_score(y_pred, y_test)) # Accuracy score

print("Data Visualization: ")  # Show data visualization
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))  # Print Confusion matrix

nclusters = 2 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(X)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(X)
score = metrics.silhouette_score(X, y_cluster_kmeans)
print("K-Means Result: ")
print("Silhouette Score of K-Means: ", score)
