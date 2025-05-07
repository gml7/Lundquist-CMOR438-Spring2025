# Single Neurons
A single neuron in machine learning is a basic unit that mimics how a brain cell processes information. It takes in one or more input values, multiplies each by a weight (which shows how important that input is), adds them up, and then passes the result through an activation function to decide the output. Despite being simple on the scale of most machine learning algorithms, it can still be very useful as a way to classify low-dimensional data and fit multivariate functions. 

We cover three types of single neuron models:
### The Perceptron
The perceptron is one of the earliest and simplest types of artificial neurons used in machine learning. It takes several input values, multiplies each by a weight, adds them up, and passes the sum through a step function (called an activation function) to produce a binary output — typically 0 or 1.

The perceptron learns by adjusting its weights based on whether its output is correct or not, gradually improving over time. It works well for problems where the data can be separated with a straight line (called linearly separable problems), but it has limitations for more complex tasks. Despite its simplicity, the perceptron laid the foundation for modern neural networks.
#### Implementation
Check the `Perceptron.ipynb` notebook for a demonstration of the perceptron on the UCI Wine dataset. The bulk of the algorithmic code may be found in the `SingleNeuron.jl` file.

### Linear regression
Linear regression is a basic statistical and machine learning method used to model the relationship between one or more input variables (features) and a continuous output (target). It works by fitting a straight line (or a flat plane in higher dimensions) through the data that best predicts the output based on the inputs.

The model calculates the output as a weighted sum of the inputs plus a bias (intercept). During training, it adjusts the weights and bias to minimize the difference between the predicted values and the actual values, typically using a method called least squares.

Linear regression is widely used for predicting things like prices, trends, or measurements when the relationship between variables is roughly linear.
#### Implementation
Check the `LinearRegression.ipynb` notebook for a demonstration of the perceptron on the UCI Wine dataset. The bulk of the algorithmic code may be found in the `SingleNeuron.jl` file.

### Logistic regression
Logistic regression is a machine learning algorithm used for binary classification: predicting one of two possible outcomes (like yes/no or spam/not spam). It works by using a weighted sum of the input features, then applying a sigmoid function to squeeze the output into a probability between 0 and 1. If the probability is above a certain threshold (like 0.5), it predicts one class; otherwise, it predicts the other. 
#### Implementation
Check the `LogisticRegression.ipynb` notebook for a demonstration of the perceptron on the UCI Wine dataset. The bulk of the algorithmic code may be found in the `SingleNeuron.jl` file.
