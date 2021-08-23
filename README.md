# ML-Algorithms
Visualization ML's Algorithms

I) Algorithm K-Means:
1) Definition: 
- Iterative algorithm that tries to partition the dataset into K pre-defined distinct non-overlapping subgroups (clusters) where each data point belongs to only one group. It tries to make the intra-cluster data points as similar as possible.
2) Mathematical Basis: 
- It assigns data points to a cluster such that the sum of the squared distance between the data points and the cluster's centroid (arithmetic mean of all the data points that belong to that cluster) is at the minimum.
- The less variation we have within clusters, the more homogeneous (similar) the data points are within the same cluster.
3) Algorithm:
- Specify number of clusters K;
- Intialize centroids by first shuffling the dataset and then randomly selecting K data points for the centroids without replacement;
- Keep iterating until there is no change to centroids:
  + Compute the sum of the squared distance between data points and all centroids;
  + Assign each data point to the closet cluster (centroid);
  + Compute the centroids for the clusters by taking the average of the all data points that belong to each cluster.
4) Visualization:
![kmeans visualization](https://user-images.githubusercontent.com/63906418/130402617-7e3e91ff-5c3b-4220-85e0-68e9b9744493.png)

II) Applied K-Means for analyzing pictures:
1) Definition:
- Each picture have a lot of small squares, that we call it "pixel", 1 pixel is represented by 3 colors in color system RGB (Red-Green-Blue). The range of colors in system RGB start 0 to 255.
- We have known that, 1 pixel equals 1 vector in 3D-dimension.

2) Algorithm:
- Using library and read the original picture (function "imread";
- Reshape type of picture to a list of vector;
- Using K-Means algorithm in order to predict each pixel, which clusters belong to;
- Create a blank picture and replace the result, which received by K-Means algorithm to new picture.

3) Visualization:
- Original picture:

![a](https://user-images.githubusercontent.com/63906418/130405761-bdd952c7-314b-449f-9ecd-2b265fbe86bf.JPG)

- Picture after using K-Means Algorithm:

![Kmeans(K=4)](https://user-images.githubusercontent.com/63906418/130405603-55e03a62-5462-4f50-a0fa-4381ef643a31.png)

III) Algorithm Linear Regression:
1) Definition:
- Linear Regression is a algorithm based on supervised learning, which performs regression task. Regression models a target prediction value based on independent variables.

2) Mathematical Basis:
- Using linear algebra

3)Visualization:
- Line graph:

![linear_regression (line)](https://user-images.githubusercontent.com/63906418/130412790-84cc2367-3e4c-4633-92c2-26594c0d9db1.png)

- Parabol graph:

![linear_regression(parabol)](https://user-images.githubusercontent.com/63906418/130412831-6bfcb845-782b-4035-8f7c-fa70ea21ef58.png)

IV) Gradient Descent:
1) Definition:
- Gradient Descent is used when training data models, can be combined with every algorithm and is easy to understand and implement. He is the most popular optimization strategy.

2) Mathematical Basis:
- The most common approach is to start from a point that we consider close to the solution of the problem, and then use an iterative operation to progress to the desired point, i.e., until the derivative is close to zero.

3) Visualization:

![GD_for_Linear_Regression](https://user-images.githubusercontent.com/63906418/130415431-9241fa7f-00f3-4cad-a73a-3dec9490b4ba.png)

V) K Nearest Neighbor:
1) Definition:
- K Nearest Neighbor is a algorithm that stores all the availables case and classifies the new data or cases based on a similarity measure. 

