'''
#Introduction to K-Means++
In the traditional K-Means algorithms, the starting postitions of the centroids are intialized completely randomly. 
This can result in suboptimal clusters. K-Means++ changes the way centroids are initalized to try to fix this problem.
Step 1 of the K-Means algorithm is “Place k random centroids for the initial clusters”. The K-Means++ algorithm 
replaces Step 1 of the K-Means algorithm and adds the following:

    1.1 The first cluster centroid is randomly picked from the data points.
    1.2 For each remaining data point, the distance from the point to its nearest cluster centroid is calculated.
    1.3 The next cluster centroid is picked according to a probability proportional to the distance of each point to
    	its nearest cluster centroid. This makes it likely for the next cluster centroid to be far away from the already
    	initialized centroids.

Repeat 1.2 - 1.3 until k centroids are chosen.


#K-Means++ using Scikit-Learn
There are two ways and they both require little change to the syntax:

	1. You can adjust the parameter to init='k-means++'.
	2. Simply drop the parameter.

This is because that init=k-means++ is actually default in scikit-learn.
'''

model = KMeans(n_clusters=k, init='k-means++')

model = KMeans(n_clusters=k)

