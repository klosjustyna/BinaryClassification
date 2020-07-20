import numpy as np
import pandas as pd 
import csv
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection, svm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, cross_val_predict,cross_val_score 
from sklearn.metrics import recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors, metrics, svm, linear_model, tree
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
import sys
np.set_printoptions(threshold=sys.maxsize)
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import random
from sklearn.neural_network import MLPClassifier

# table with count of pconfirmed cases by state
state_confirmed_cases = pd.read_csv("us-states.csv")
state_confirmed_cases=state_confirmed_cases.reindex(columns=['state','confirmed_cases'])


# table with count of people by state
countOfCitizen=pd.read_csv("population-state.csv")
countOfCitizen=countOfCitizen.reindex(columns=['population'])

data = pd.concat([state_confirmed_cases, countOfCitizen], axis=1, sort=False)

# threshold infectivity   
data['infectivity'] =  data['confirmed_cases']/data['population']*100

# determine class
data['class'] = np.where(data['infectivity']>1, '1','0')  

# table with all statistic
statistic=pd.read_csv("statistic.csv")

# concat table statistic with class
data = pd.concat([data.reindex(columns=['class']), statistic], axis=1, sort=False)


averageIQ_mean = np.float32(data.averageIQ.mean())
# print((averageIQ_mean))

# fill the empty data
# print(data.describe())
data['AirQualityIndex'] = data['AirQualityIndex'].replace(np.nan,0)
data['averageIQ'] = data['averageIQ'].replace(np.nan,100.37)


# data.astype({'averageIQ': 'float32'}).dtypes
# data['averageIQ'] = data.fit_transform(data['averageIQ'])
# print (data['averageIQ'])
data.fillna(data.mean(), inplace=True)

data.values
target = ['houseIncome','density','WhitePerc','BlackPerc',
 'AirQualityIndex','totalHomeless', 'averageIQ']
X = data[target]
y = data['class']

# standardization
X = StandardScaler().fit_transform(X)
# print(X)

#  initializze PCA 
pca = PCA(0.95)
X = pca.fit_transform(X)
# print(X)

# split data 
splits = model_selection.train_test_split(X, y, test_size=.333, random_state=0)
X_train, X_test, y_train, y_test = splits  

# normalize data
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

# crossvalidate
clf = svm.SVC(kernel='linear', C=1)


def classifier (paramClassifier):
    print(str(paramClassifier))
    t0=time.time()
    clf = paramClassifier 
    scores = cross_val_score(clf, X, y,scoring='accuracy', cv=5)
    y_pred = cross_val_predict(clf, X, y, cv=10)
    conf_mat = confusion_matrix(y, y_pred)
    print(conf_mat)
    TP = conf_mat[0,0]  
    FP = conf_mat[0,1]
    FN = conf_mat[1,0]
    TN = conf_mat[1,1]

    TPR = TP/(TP+FN)*100
    print('tpr', round(TPR,2))
    FPR = FP/(FP+TN)*100
    print('fpr',round(FPR,2))
    SPC = 100 - FPR
    print('spc',round(SPC,2))
    FDR = FP/(TP+FP)*100
    print('fdr',round(FDR,2))
    PPV = TP/(TP+FP)*100
    print('ppv',round(PPV,2))
    NPV = TN/(TN+FN)*100
    print('npv',round(NPV,2))
    F = 2 * ((PPV * TPR)/(PPV + TPR))
    print('F', round(F,2))
    acc = scores.mean()*100
    print ((round(acc,2),'%'))
    print('*****************************')
    print ("training time:", round(time.time()-t0, 5), "s" )
    return scores.mean()

def startAllClassifier():

    classifier(GaussianNB())
   
    classifier(LogisticRegression())

    classifier(tree.DecisionTreeClassifier())

    classifier(svm.SVC(probability=True))

    classifier(RandomForestClassifier(n_jobs=2, random_state=0,max_depth=4))

    classifier(AdaBoostClassifier(n_estimators=100, random_state=0))

    numberOfNeighbors = 6
    classifier(neighbors.KNeighborsClassifier(n_neighbors=numberOfNeighbors))

    classifier(MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500))

startAllClassifier()

def kNN():
    a = []
    b = []
    x = 0 
    for i  in range(2,35):
        x = classifier(neighbors.KNeighborsClassifier(n_neighbors=i))
        a.append(i)
        b.append(x)
    
    print(max(b))

    print('srednia', sum(b)/len(b))
    print('max', max(b))
    print(b)
    plt.plot(a,b)
    plt.xlabel('Ilość najbliższych sąsiadów ')
    plt.ylabel('Dokładność klasyfikatora')
    plt.title('')
    plt.show()

# kNN()

def plot1():
    plt.plot(['kNN','Naive Bayes','Drzewo decyzyjne','SVM','Regresja logistyczna','Lasy losowe','Ada Boost','Sieć neuronowa'],[78.5,	60.00,	71.11,	64.71,	68.75,	66.67,	81.82,	75.0], label='TPR')
    plt.plot(['kNN','Naive Bayes','Drzewo decyzyjne','SVM','Regresja logistyczna','Lasy losowe','Ada Boost','Sieć neuronowa'],[100,	0.00,	33.33,	0.00,	100,	50.0,	66.67,	60.0], label='SPC')
    plt.plot(['kNN','Naive Bayes','Drzewo decyzyjne','SVM','Regresja logistyczna','Lasy losowe','Ada Boost','Sieć neuronowa'],[0.00,	100,	66.67,	0.00,	0.00,	50.0,	33.33,	40.0], label='FPR')
    plt.plot(['kNN','Naive Bayes','Drzewo decyzyjne','SVM','Regresja logistyczna','Lasy losowe','Ada Boost','Sieć neuronowa'],[0.00,	18.18,	11.11,	0.00,	0.00,	9.09,	18.18,	18.18,], label='FDR')
    plt.plot(['kNN','Naive Bayes','Drzewo decyzyjne','SVM','Regresja logistyczna','Lasy losowe','Ada Boost','Sieć neuronowa'],[100,	81.82,	88.89,	100,	100,	90.91,	81.82,	81.82], label='PPV')
    plt.plot(['kNN','Naive Bayes','Drzewo decyzyjne','SVM','Regresja logistyczna','Lasy losowe','Ada Boost','Sieć neuronowa'],[50.00,0.00,13.33,0.00,16.67,16.67,66.67,50.0], label='NPV')
    plt.plot(['kNN','Naive Bayes','Drzewo decyzyjne','SVM','Regresja logistyczna','Lasy losowe','Ada Boost','Sieć neuronowa'],[88.00,69.23,76.19,78.57,81.48,76.92,81.82,78.26], label='F')

    plt.xlabel('Algorytm')
    plt.ylabel('Miara')
    plt.legend();
    plt.show()

# plot1()

print ((72.55	+64.55	+55.09	+70.55	+70.36	+60.55	+68.73	+68.55)/8)

def accurancyPlot():
    bars = ['kNN','Naive Bayes','Drzewo decyzyjne','SVM','Regresja logistyczna','Lasy losowe','Ada Boost','Sieć neuronowa']
    value = [0.0319,	0.01895,	0.02593,	0.02596,	0.06685,	4.06115,	3.27228,	0.35266]
    y_pos = np.arange(len(value))
    
    #Figsize
    plt.figure(figsize=(10,5))
    
    # Create bars
    plt.bar(y_pos, value, color='#81BEF7')
    
    # Create names on the x-axis
    plt.xticks(y_pos, bars)
    
    plt.xlabel('Klasyfiaktor', fontsize=12, color='#323232')
    plt.ylabel('Dokładnośc', fontsize=12, color='#323232')
    axes = plt.gca()
    plt.yticks(np.arange(0, 5, 0.5))
    plt.show()


accurancyPlot()    


