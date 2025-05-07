# Neural Networks 
A neural network is a collection of many single neurons connected together in layers. Each neuron processes input by combining it with weights, applying an activation function, and passing the result to the next layer. The network usually has:
    - An input layer to receive the raw data
    - One or more hidden layers where neurons work together to detect patterns
    - An output layer that produces the final prediction

As data flows through the network, each neuron contributes to identifying features or patterns. During training, the network adjusts the weights in its neurons to improve accuracy. By combining many simple neurons, neural networks can learn to perform complex tasks like recognizing faces, translating languages, or playing games.

## Considerations
Overfitting is a concern in supervised learning. Overfitting is when a machine learning model learns the training data too well, including its noise and details, which makes it perform poorly on new, unseen data. But clearly to assess if a model works, one needs labeled data. A common approach is to split labeled data into a training set and a testing set. You train the model on the training set, then use the error in the model's performance on the testing set to assess the accuracy of the model.

## Implementation
Check the `NeuralNetwork.ipynb` notebook for a demonstration of a neural network on the FashionMNIST dataset.