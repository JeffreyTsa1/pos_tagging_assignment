import string
import scipy
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('always')  # This makes it so that each warning is only shown once. 

from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import OneHotEncoder
# I tried using TfidVectorizer and OneHotEncoder but didn't have as much success with them.

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


# Take the data as input, return two compressed sparse row matrices for processing.
def encode(source):
	le = LabelEncoder()
	center = np.array(source['a4'])
	classes = le.fit_transform(source[['class']])
	class_inverted = le.inverse_transform((classes))
	class_map = {}

	for i in range(len(classes)):
	    if class_inverted[i] not in class_map:
	        class_map[class_inverted[i]] = [classes[i], 1]
	    else:
	        class_map[class_inverted[i]][1] += 1
	
	lowest = float('inf') #Set it to a theoretical upper bound
	lowest_class = ""
	
	for key in class_map: #Find the lowest
		if class_map.get(key)[1] < lowest:
			lowest = class_map.get(key)[1]
			lowest_class = class_map.get(key)[0]
	
			print(class_map)
			print(class_map.get(key)[1])
	
	word_dict = {}
	final_list = []

	for i in range(len(center)):
		if center[i] not in word_dict:
			word_dict[center[i]] = classes[i]

	selected_attributes = source[['a1', 'a2', 'a3', 'a4', 'a5', 'a6']].to_numpy() #Use the leave one out method here by removing attributes.

	for row in selected_attributes: 
		local_list = []
		for i in row:
			if i in word_dict:
				local_list.append(word_dict.get(i))
			else:
				local_list.append(lowest_class)
		final_list.append(local_list)


	x = scipy.sparse.csr_matrix(final_list)
	y = scipy.sparse.csr_matrix(classes)

	return x,y #return two compressed sparse row matrices with unique value IDs for each class.


# Helper function for training
def trainer(classifier, x_train, x_test, y_train, y_test):


    classifier.fit(x_train.todense(), y_train)
    y_pred = classifier.predict(x_test.todense())


    f1 = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro", labels=np.unique(y_pred)) #Unique labels that fixes a y_pred error
    recall = recall_score(y_test, y_pred, average="macro", labels=np.unique(y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    return f1, precision, recall, accuracy

# Function for training the Decision Tree Classifier
def trainDTC(x,y):
	f1List = list()
	precisionList = list()
	recallList = list()
	accuracyList= list()
	
	# n_splits = 5, we are using 5-fold cross validation
	kf = KFold(n_splits = 5)

	for train_index, test_index in kf.split(x):
		x_train, x_test = x[train_index], x[test_index]
		y_train = []
		y_test = []
		for i in train_index:
			y_train.append(y[0, i])
		for i in test_index:
			y_test.append(y[0, i])
		f1, precision, recall, accuracy= trainer(DecisionTreeClassifier(random_state=0), x_train, x_test, y_train, y_test)
		f1List.append(f1)
		precisionList.append(precision)
		recallList.append(recall)
		accuracyList.append(accuracy)


	averageAccuracy = np.mean(accuracyList)
	averagePrecision = np.mean(precisionList)
	averageF1 = np.mean(f1List)
	averageRecall = np.mean(recallList)
	print("Decision Tree Classifier Average Accuracy for: {}".format(averageAccuracy))
	print("Decision Tree Classifier Average Precision for: {}".format(averagePrecision))
	print("Decision Tree Classifier Average F1 for: {}".format(averageF1))
	print("Decision Tree Classifier Average Recall for: {}".format(averageRecall))

# Function for training the Decision Tree Classifier
def trainNB(x,y):
	f1List = list()
	precisionList = list()
	recallList = list()
	accuracyList= list()
	
	# n_splits = 5, we are using 5-fold cross validation
	kf = KFold(n_splits = 5)

	for train_index, test_index in kf.split(x):
		x_train, x_test = x[train_index], x[test_index]
		y_train = []
		y_test = []
		for i in train_index:
			y_train.append(y[0, i])
		for i in test_index:
			y_test.append(y[0, i])
		f1, precision, recall, accuracy= trainer(GaussianNB(), x_train, x_test, y_train, y_test)
		f1List.append(f1)
		precisionList.append(precision)
		recallList.append(recall)
		accuracyList.append(accuracy)


	averageAccuracy = np.mean(accuracyList)
	averagePrecision = np.mean(precisionList)
	averageF1 = np.mean(f1List)
	averageRecall = np.mean(recallList)
	print("Naive Bayes Average Accuracy for: {}".format(averageAccuracy))
	print("Naive Bayes Average Precision for: {}".format(averagePrecision))
	print("Naive Bayes Average F1 for: {}".format(averageF1))
	print("Naive Bayes Average Recall for: {}".format(averageRecall))

# Run!
def build():
	data = pd.read_csv('pos-eng-5000.data.csv')
	print(data)
	x, y = encode(data)
	trainNB(x,y)
	trainDTC(x,y)


build()