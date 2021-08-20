import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img = plt.imread("a.JPG")

height = img.shape[0]
width = img.shape[1]

img = img.reshape(width*height,3)

kmeans = KMeans(n_clusters=4).fit(img) #Using elbow method to find how many clusters

labels = kmeans.predict(img)
clusters = kmeans.cluster_centers_ #The average value of color according to clusters (RGB)

#Case 1: Create a string and then reshape to new picture
"""
img2 = np.zeros_like(img) #Create a 0 numpy array like variable img

for i in range(len(img2)): #loop for each array in img2
    img2[i] = clusters[labels[i]] #Replace value in each array in img2 by value color RGB in cluster

img2 = img2.reshape(height,width,3) #Reshape array to draw a new pic

plt.imshow(img2)
plt.show()
"""

#Case 2: Create a picture with the size of existence pic and the change each value that receive a new pic 
img2 = np.zeros((height,width,3), dtype=np.uint8)

#Horizontal loop
index = 0
for i in range(height):
    for j in range(width):
        label_of_pixel = labels[index]
        img2[i][j] = clusters[label_of_pixel]
        index += 1

plt.imshow(img2)
plt.show()