import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# shape : should see (150, 5) meaning 150 instances, 5 attributes
#print(dataset.shape)
# head : should see first twenty rows of the data
#print(dataset.head(20))
# describe : summary of each attribute
#print(dataset.describe())
# groupby, size : can look at number of instances (rows) that belong to each class
#print(dataset.groupby('class').size())
