#Iris Flowers Machine Learning Dataset
from sklearn import datasets, neighbors
import numpy as np
import math
import operator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Building KNN algorithm without using exist library 
def cal_distance(p1,p2): #Find the distance between p1,p2
    dim = len(p1) #Dimesion 
    distance = 0
    for i in range(dim):
        distance += (p1[i]-p2[i])*(p1[i]-p2[i])
    return math.sqrt(distance) 

def get_k_neighbors(training_X, label_y, point,k): #Find K-points nearest the input-point
    distances = [] #List, which include the distance between input-point and each point in training_X
    neighbors = [] #List, which include K-points min and correlative label

    #Calculate distance from point to everything in training set X
    for i in range(len(training_X)):
        distance = cal_distance(training_X[i],point)
        distances.append(distance)
        #distances.append((distance,label_y[i])) using tuple to find K-points min
    
    #distances.sort(key=operator.itemgetter(0)) #sort by distance using library
    
    #Position of k smallest distance
    index = [] 

    #Get k closet points
    while len(neighbors) < k: 
        i = 0
        min_distance = 999999
        min_idx = 0
        while i < len(distances):
            #Skip the nearest points that have been counted
            if i in index:
                i+=1
                continue
            
            #Update smallest distance and index 
            if distances[i] <= min_distance:
                min_distance = distances[i]
                min_idx = i
            i+=1

        #Add min index so we skip it in the next iteration
        index.append(min_idx)
        neighbors.append(label_y[min_idx])
    """   
    for i in range(k):
        neighbors.append(distances[i][1])
    """
    return neighbors

def highest_votes(labels): #Popular labels 
    labels_count = [0,0,0]
    for label in labels:
        labels_count[label] +=1

    max_count = max(labels_count)
    return labels_count.index(max_count)

def predict(training_X, label_y,point, k):
    neighbors_labels = get_k_neighbors(training_X, label_y, point, k)
    return highest_votes(neighbors_labels)

def accuracy_score(predicts, labels):
    total = len(predicts)
    correct_count = 0
    for i in range(total):
        if predicts[i] == labels[i]:
            correct_count += 1

    accuracy = correct_count/total
    return accuracy

iris = datasets.load_iris()
iris_X = iris.data #data (petal length, petal width, sepal length, sepal width)
iris_Y = iris.target #label 
iris_Y = np.array(iris_Y)

#Shuffle by index
rand_Index = np.arange(iris_X.shape[0])
np.random.shuffle(rand_Index)

iris_X = iris_X[rand_Index]
iris_Y = iris_Y[rand_Index]

#Divide data into 2 groups: 100 (training_set), 50 (test_set)
"""
X_train = iris_X[:100,:]
X_test = iris_X[100:,:]
y_train = iris_Y[:100]
y_test = iris_Y[100:]

k = 5
y_predict = []
for p in X_test:
    label = predict(X_train, y_train, p, k)
    y_predict.append(label)

print(y_predict)
print(y_test)

acc = accuracy_score(y_predict, y_test)
print(acc)
"""

#Using KNN from library scikit-learn
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_Y, test_size=50)

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
acc = accuracy_score(y_predict, y_test)

print(y_predict)
print(y_test)
print(acc)