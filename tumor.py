# import the python sk learn library for scikitlearn
# this library is built on NumpY, Matplotlib and sciPy - foundation libraries for python machine learning
# you must install these libraries themselves, run code below on terminal
# pip install numpy matplotlib scipy scikit-learn
import sklearn

# find the breast cancer winsconsin diagnostic dataset in the sklearn dataset libraries
from sklearn.datasets import load_breast_cancer 

#now store the imported breast cancer data set into a variable for easy reuse
cancer_data = load_breast_cancer()

# note this data set is in form of dictionary data type
#print the keys
print( cancer_data.keys())

# you will notice some keys like the target names  == classification label
# shows results as malignant or beningn --- because the values here are not singular values but array list values of only 2 items
cancer_label_names = cancer_data["target_names"]
print(cancer_label_names)

# next is the target - actual truth labels for each data 0 for malignant 1 for benign
# this is basically the correct answers to my data. i use it to teach the ai (data index 1 == target index 1)
cancer_labels = cancer_data["target"]
print(cancer_labels)

# this one shows all the information collected for each patient in the data
cancer_feature_names = cancer_data["feature_names"]
print(cancer_feature_names)

# the actual data to train, each row represents each of the 569 patients data
cancer_features = cancer_data["data"]
print(cancer_features)

# aftyer collecting the data, we need to organzie the dat into training set and testing set
# sklearn has fucntion to split your data into 2 places
from sklearn.model_selection import train_test_split

#now invoke the train_test_split fucntion to specify what percentage should be used for testing and what percentage for training
#this function has 4 parameters (for the train set  - train data, train label and for test set- we need test data and test label)
# view the function teh first paam is a *array -- which is where my variables collects from the 2d nparray daTASET OF BREATS CANCER

#the test data can be highly imbalanced and ill learn the ai. say you have a dataset and about 70 % is benign and 30 percent is malignanet
# if u split that data say into say 0.3 for testing . the sklearn might randomly select only the 70 percent benign as its train data and leave the rest for testing
# this causes bias as the ai didnt have much data to train on malignants hence the results might still be skewed towards malignant during test
# with startify we ensure that the ratio split is applied in the randomly picked data set, say i split it 60/40, it will ensure that train data has 60 bening and 40 malignant
# then also the train is also 60/40 like that too, that way the ai has seen enough of booth cases to learn properly

X_train, X_test, y_train, y_test = train_test_split(cancer_features, cancer_labels, test_size=0.2, random_state=20, stratify=cancer_labels)

# show the train data amoutnt - (a, b) a=no of samples i.e patient, b= no of features for each sample
print("X_train shape:", X_train.shape)

#test data
print("X_test shape:",  X_test.shape)

#just show no of samples
print("y_train shape:", y_train.shape)
print("y_test shape:",  y_test.shape)

# u can apply Naive Bayes Algortihm. This naives baives allows sklearn to make very naive assumptions
# it follows the baye's theorem; it assumes that the data fed to it is independent of each other
# can reduce accuracy but very insignificant at the profit of faster predictions
# import the naives bayes package from sklearn and the gaussianNB class
from sklearn.naive_bayes import GaussianNB

#store that class in a variable for ease of re use
gnb = GaussianNB()

#NOW create an ai model using this module and feed it the data for training
# it has a fit method that takes the train and train labels datas for training
# store it in a different valriable model which basically is the trained version of ur ai
model = gnb.fit(X_train, y_train)

#make predictions randomly from ur test data portion using the trained version of ur ai
predictions = model.predict(X_test)

# display predictions for all test data
print(predictions)

# display prediction on a particular sample in the test data
print(predictions[3])

# you can check the accuracy of your ai, since this is supervised we already know what the test data actual truths are
# comparing this predictions with the actual truths for the test data
# impoort the metrics package of sk learn to use the accuracy score function(truth, prediction)
from sklearn.metrics import accuracy_score, confusion_matrix

# calculate it in percentage
percentage_accuracy = (accuracy_score(y_test, predictions) ) * 100

# display the accuracy score in percent
print("the percenatge accuracy of your ai model is " , percentage_accuracy)

# to know the exact no of smaples were correct or wrong, use the confusion matrix function of sklearn
# see line 87
cm = confusion_matrix(y_test, predictions)

#diusplay the confusion matrix
print("The confusion matrix is given as follows\n" , cm)

# results will look like this 
#       [a      ,     b]  -> this row is for beningn
#       [c      ,     d]   -> row for malignants
#   a = correctly predicted beningn
#   b = incorrect in beningn. predicted malignant instead of benign
#   c = incorrect preiction of malignant. predicted benign instead of malignant - dangerous
#   d = correctly predicted malignant

