#        -- TODO --
# - x-fold cross validation
# - dimensionality reduction
# - alternative weighting
# - adding features
#   - pinpoint emojis
#   - differentiate between turns
#   - POS, NER
#   - find emotional words
# - other classifiers
#   - Randomize baseline
#   - Naive Bayes baseline
#   - Random forest
#   - Neural Network
#   - different SVM kernel

# DEPENDENCIES: sklearn, pandas, pandas_ml, spacy

import re, sys
from time import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from pandas_ml import ConfusionMatrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# import spacy
# import en_core_web_sm
# Spacy model setup
# nlp = en_core_web_sm.load()

# Taking arg of file name, if no arg given, assumes 
file_name = 'train.txt'
if len(sys.argv) > 1:
    file_name = sys.argv[1]

# reading training file into dataframe
print("Reading train file")
t = time()
instances = pd.read_csv('train.txt', sep='\t', header=0)
print("Finished reading train file in %0.3fsec\n" % (time()-t))

# Separating the labels and strings into separate arrays & concatenating turns from bag of words
print("Separating text and labels")
t = time()
row_strings = []
labels = []
for index, instance in instances.iterrows():
    row_strings.append(instance['turn1'] + ' ' + instance['turn2'] + ' ' + instance['turn3'])
    labels.append(instance['label'])
print("Separating text and labels finished in %0.3fsec\n" % (time()-t))

# 60/40 split of training and test data
X_train, X_test, y_train, y_test = train_test_split(row_strings, labels, test_size=0.4, random_state=0)

# Creating bag of words feature vectors from training data
print("Creating training feature vectors")
t = time()
# countBoW = CountVectorizer() # add stopword removal or ngrams with: ngram_range=(1,2), stop_words='english'
tfidfBoW = TfidfVectorizer(ngram_range=(1,2))
X_train_vec = tfidfBoW.fit_transform(X_train)
print("Creating feature vectors finished in %0.3fsec\n" % (time()-t))

# print (pd.DataFrame(data=X_train_vec.toarray(), columns=tfidfBoW.get_feature_names()))

# svd = TruncatedSVD(n_components=300) #use sparse setting instead?
# pca = PCA(n_components=300)

# # Training Naive Bayes on training data
# print("Training Naive Bayes")
# t = time()
# nbClassifier = GaussianNB(priors=None, var_smoothing=1e-09).fit(X_train_vec, y_train)
# print("Training Naive Bayes finished in %0.3fsec\n" % (time()-t))

# # Training Random Forest on training data
# print("Training Random Forest")
# t = time()
# rfClassifier = ___.fit(X_train_vec, y_train)
# print("Training Random Forest finished in %0.3fsec\n" % (time()-t))

# Training linear kernel SVM on training data
print("Training SVM")
t = time()
svmClassifier = svm.SVC(kernel='linear', C=1).fit(X_train_vec, y_train)
print("Training SVM finished in %0.3fsec\n" % (time()-t))


# Creating bag of words feature vectors from test data
print("Vectorizing test features")
t = time()
X_test_vec = tfidfBoW.transform(X_test)
print("Vectorizing test features finished in %0.3fsec\n" % (time()-t))

# Creating confusion matrix
print("Predicting test instances\n")
y_true = y_test
y_pred = svmClassifier.predict(X_test_vec)
confusion_matrix = ConfusionMatrix(y_true, y_pred)

print("Confusion matrix:\n%s\n" % confusion_matrix)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print("Accuracy: %s\nPrecision: %s\nRecall: %s\nF1-Score: %s\n" % (accuracy, precision, recall, f1))

