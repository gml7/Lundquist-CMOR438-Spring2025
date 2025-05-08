# DBSCAN
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is an unsupervised machine learning algorithm used for clustering data based on density. Unlike algorithms like K-means, DBSCAN can find arbitrarily shaped clusters and automatically detects noise (outliers).

Given a distance metric and some distance $\epsilon$, we count the number of data points within $\epsilon$ to determine the density at a particular point. Outliers, then, are data points with a low density region around them. A *core point* is a point within $\epsilon$ of at least $k$ other points.

### Hyperparameters
$\epsilon, k$ are hyperparameters.

### Algorithm
1. Identify core points
2. Set `c = 1`
3. If there are no core points `p` not already in a cluster, label the rest of the points as anomalies and STOP.
4. Otherwise, randomly select a core point `p` not already in a cluster and assign it to cluster `c`
5. Add all core points (and only core points) within $\epsilon$ of `p` to cluster `c`
6. For all core points `q` within `c`:
	1. Add all core points within $\epsilon$ of `q` to cluster `c`
7. If there are still core points within $\epsilon$ of any core points `q` within `c`, go to step 5
8. Otherwise, add to `c` all *non*-core points within $\epsilon$ of any point `p` within `c` and continue
9. Set `c = c + 1`
10. Go to step 3

### Considerations
Finding appropriate $\epsilon, k$ takes nontrivial experimentation, and the performance of the algorithm depends heavily on those values. DBSCAN can also struggle with clusters of varying densities.

