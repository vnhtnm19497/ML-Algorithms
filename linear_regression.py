import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model 


#Random data
A = [2,5,7,9,11,16,19,23,22,29,29,35,37,40,46]
b = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

#Visualize data (Linegraph)
plt.plot(A,b, 'ro')

A = np.array([A]).T #Transpose A
b = np.array([b]).T #Transpose b

#Create vector 1 
vector_ones = np.ones((A.shape[0],1), dtype = np.int8)

#Combine A and vector 1
A = np.concatenate((A, vector_ones), axis=1) 

#Use formula
x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b)

#Test data to draw
x0 = np.array([[1,46]]).T
y0 = x[0][0]*x0 + x[1][0] 

#Visualize data (Parabol graph)
A1 = [2,5,7,9,11,16,19,23,22,29,29,35,37,40,46,42,39,31,30,28,20,15,10,6]
b1 = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

plt.plot(A1,b1,'ro')

A1 = np.array([A1]).T #Transpose A1
b1 = np.array([b1]).T #Transpose b1

#Create A1 square
x_square = np.array([A1[:,0]**2]).T
A1 = np.concatenate((x_square, A1), axis =1) #Combine x^2 and A1

#Create vector 1 
vector_ones = np.ones((A1.shape[0],1), dtype=np.int8)

#Combine A1 and 1
A1 = np.concatenate((A1, vector_ones), axis=1) 

#Use formula 
x = np.linalg.inv(A1.transpose().dot(A1)).dot(A1.transpose()).dot(b1)

#Test data to draw
x0 = np.linspace(1,46,10000)
y0 = x[0][0]*x0*x0 + x[1][0]*x0 + x[2][0]

plt.plot(x0,y0)
plt.show()

"""
#Using library Scikit Learn
#Random data
A = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,37,40,46]]).T
b = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]).T

#Create model
lr = linear_model.LinearRegression()

lr.fit(A, b) #Train data
print(lr.intercept_)
print(lr.coef_) 

plt.plot(A,b, 'ro')

x0 = np.array([[1,46]]).T
y0 = lr.coef_*x0 + lr.intercept_ 

plt.plot(x0,y0)
plt.show()
"""