# K-Means Clustering
K-means clustering is an unsupervised machine learning algorithm used to group similar data points into K distinct, non-overlapping clusters. It aims to partition a dataset into K clusters in which each data point belongs to the cluster with the nearest mean (centroid).

How it works:
1. Initialization: Choose the number of clusters K and randomly initialize K centroids.
2. Assignment: Assign each data point to the nearest centroid, forming K clusters.
3. Update: Recalculate the centroids as the mean of all data points in each cluster.
4. Repeat: Steps 2 and 3 are repeated until centroids no longer change significantly or a maximum number of iterations is reached.

Key Features:
- Distance metric: Typically uses Euclidean distance to measure closeness.
- Centroid-based: Clusters are defined by the center (mean) of the points in them.
- Iterative: The algorithm refines clusters with each iteration.
- Fast and efficient: Works well with large datasets, but requires the user to specify K in advance.

Limitations:
- Sensitive to the initial placement of centroids.
- Not ideal for non-spherical or overlapping clusters.
- The value of K must be chosen beforehand.

## Implementation
The algorithm is explained and implemented in the `KMeansClustering.ipynb` jupyter notebook. Run the cells in order to reproduce our results.