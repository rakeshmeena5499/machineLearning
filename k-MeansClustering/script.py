'''
Clustering is the most well-known unsupervised learning technique. It finds structure in unlabeled data by identifying
similar groups, or clusters. Examples of clustering applications are:

1. Recommendation engines: group products to personalize the user experience
2. Search engines: group news topics and search results
3. Market segmentation: group customers based on geography, demography, and behaviors
4. Image segmentation: medical imaging or road scene segmentation on self-driving cars

'''

import codecademylib3_seaborn
import matplotlib.pyplot as plt

from sklearn import datasets

iris = datasets.load_iris()

print(iris.data)

print(iris.target)

print(iris.DESCR)


'''
#Implementation of K-Means Clustering

        1. Place k random centroids for the initial clusters.
        2. Assign data samples to the nearest centroid.
        3. Update centroids based on the above-assigned data samples.

'''

import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from copy import deepcopy

iris = datasets.load_iris()

samples = iris.data

x = samples[:,0]
y = samples[:,1]

sepal_length_width = np.array(list(zip(x, y)))

# Step 1: Place K random centroids

k = 3

centroids_x = np.random.uniform(min(x), max(x), size=k)
centroids_y = np.random.uniform(min(y), max(y), size=k)

centroids = np.array(list(zip(centroids_x, centroids_y)))

def distance(a, b):
    one = (a[0] - b[0]) ** 2
    two = (a[1] - b[1]) ** 2
    distance = (one + two) ** 0.5
    return distance

# To store the value of centroids when it updates
centroids_old = np.zeros(centroids.shape)

# Cluster labeles (either 0, 1, or 2)
labels = np.zeros(len(samples))

distances = np.zeros(3)

# Initialize error:
error = np.zeros(3)

error[0] = distance(centroids[0], centroids_old[0])
error[1] = distance(centroids[1], centroids_old[1])
error[2] = distance(centroids[2], centroids_old[2])

# Repeat Steps 2 and 3 until convergence:

while error.all() != 0:

    # Step 2: Assign samples to nearest centroid

    for i in range(len(samples)):
        distances[0] = distance(sepal_length_width[i], centroids[0])
        distances[1] = distance(sepal_length_width[i], centroids[1])
        distances[2] = distance(sepal_length_width[i], centroids[2])
        cluster = np.argmin(distances)
        labels[i] = cluster

    # Step 3: Update centroids

    centroids_old = deepcopy(centroids)

    for i in range(3):
        points = [sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i]
        centroids[i] = np.mean(points, axis=0)

    error[0] = distance(centroids[0], centroids_old[0])
    error[1] = distance(centroids[1],   centroids_old[1])
    error[2] = distance(centroids[2], centroids_old[2])

colors = ['r', 'g', 'b']

for i in range(k):
    points = np.array([sepal_length_width[j] for j in range(len(samples)) if labels[j] == i])
    plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.5)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=150)

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')

plt.show()


'''
#Implementing K-Means: Scikit-Learn
'''

import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn import datasets

# From sklearn.cluster, import KMeans class
from sklearn.cluster import KMeans

iris = datasets.load_iris()

samples = iris.data

# Use KMeans() to create a model that finds 3 clusters
model = KMeans(n_clusters = 3)
# Use .fit() to fit the model to samples
model.fit(samples)
# Use .predict() to determine the labels of samples
labels = model.predict(samples)
# Print the labels
print(labels)
