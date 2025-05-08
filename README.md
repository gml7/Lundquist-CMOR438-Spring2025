# CMOR 438 / INDE 577 Data Science & Machine Learning

This repository houses code illustrating various machine learning concepts and algorithms taught during the spring 2025 semester of CMOR 438 at Rice University. 

### Author
Gabriel M. Lundquist

## Concepts

### Supervised Learning
Supervised learning is a type of machine learning where a model is trained using labeled data. That means each example in the training set comes with the correct answer. The goal is for the model to learn patterns so it can make accurate predictions on new, unseen data.

We cover the use of supervised learning in both regression and classification, and the particular concepts/algorithms we cover are:
- Single Neuron models
- Neural networks
- K-Nearest Neighbor

### Unsupervised Learning
Unsupervised learning identifies clusters and trends in unlabeled data. The goal is unchanged, but the methods must be different.

We cover the use of unsupervised learning in classification, in particular:
- K-Means Clustering
- Density-Based Clustering


## The Case for Julia
The algorithms are implemented and demonstrated in Julia. Julia is a scientific computing language whose creators pose it as a more efficient but equally intuitive alternative to Python & Numpy. Julia is well-documented, functional, and occasionally even more convenient than Python, incorporating vectorized operations. With a few optimization considerations it can get close to metal. Both speed and vectorization are important for machine learning. If nothing else, I found a good excuse to learn a new language. 