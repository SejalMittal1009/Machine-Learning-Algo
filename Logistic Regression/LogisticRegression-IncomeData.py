import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn import linear_model
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


warnings.filterwarnings("ignore")


# Input file containing data
input_file = 'income_data.txt'

# Read the data
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break

        if '?' in line:
            continue

        data = line[:-1].split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1

        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Convert to numpy array
X = np.array(X)

# Convert string data to numerical data
label_encoder = [] 
X_encoded = np.empty(X.shape)
for i,item in enumerate(X[0]):
    if item.isdigit(): 
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Create SVM classifier
#classifier = OneVsOneClassifier(LinearSVC(random_state=0))

# Train the classifier
#classifier.fit(X, y)
plt.figure()
# Cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
w=[]
v=[]
#classifier svm
classifier = OneVsOneClassifier(LinearSVC(random_state=0))
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
f1=round(100*f1.mean(), 2)
print("F1 scores svm: " + str(f1) + "%")
w.append(f1)

#logistic regression classifer
classifier=linear_model.LogisticRegression(solver="liblinear",C=100)
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
f1=round(100*f1.mean(), 2)
print("F1 score logistic regression: " + str(f1) + "%")
w.append(f1)

#naive bayes classifer
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

# Compute the F1 score of the naive bayes classifier
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
f1=round(100*f1.mean(), 2)
print("F1 score naive bayes: " + str(f1) + "%")
w.append(f1)
v=['SVM','Logistic regression','Naive Bayes']
index=np.arange(len(v))
plt.bar(index,w)
plt.xlabel('Classifier')
plt.ylabel('F1 Score')
plt.xticks(index,v)
plt.show()


