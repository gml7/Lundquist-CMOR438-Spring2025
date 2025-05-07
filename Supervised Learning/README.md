# Supervised learning

Supervised learning is a machine learning approach where a model is trained on a dataset that includes both input data and the correct output, or "label." The model learns by comparing its predictions to the actual answers and adjusting itself to reduce errors. Over time, it learns to make accurate predictions on new, unseen data by recognizing patterns from the training examples. This method is commonly used for tasks like classification (e.g., identifying emails as spam or not) and regression (e.g., predicting house prices).

## Algorithms
Here we illustrate:
- The single neuron model, including:
    - The perceptron for classification
    - Linear regression
    - Logistic regression for classification
- Neural networks for classification
- K-nearest neighbor for classification and regression

## Implementations
How each algorithm works is illustrated in a jupyter notebook, using a Julia kernel. The single neuron models are implemented in a separate Julia file, which may be included in other projects.

## Data
We use the venerable UCI Wine dataset for the lightweight single neuron and nearest neighbor algorithms, and we use the FashionMNIST dataset to demonstrate neural networks. Each has well-defined features and targets to train on.