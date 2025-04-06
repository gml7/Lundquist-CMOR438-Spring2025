import numpy as np
from progress.bar import ShadyBar
import matplotlib.pyplot as plt

class SingleNeuron(object):
    """
    A class implementing single-neuron machine learning algorithms.
    Implements the perceptron, linear regression classification, 
    and logistic regression classification.

    ...

    Attributes
    ----------
    type_perceptron [class attribute] : string
        Contains the keyword used to indicate that the algorithm
        used by the instance is perceptron.

    type_linear_regression_1D [class attribute] : string
        Contains the keyword used to indicate that the algorithm
        used by the instance is one-dimensional linear regression.
    
    data_dimension : int
        The size of the feature vectors to be input to the model, i.e.
        the number of weights.

    model_type : str
        Contains the keyword indicating which algorithm the neuron 
        instance is using.

    activation_function : function
        A handle to the function defining how to process the 
        preactivation value.

    weights : numpy.ndarray or float
        The weights of the model.
    
    bias : float
        The bias of the model.

    previous_weights : numpy.ndarray or float
        The weights of the model previous to the most recent training.
    
    previous_bias : float
        The bias of the model previous to the most recent training.

    loss_history : list of floats
        The model's loss on the training data in each epoch of its
        training, according to the loss function used in each training.

    prev_loss_history : list of floats
        The loss history previous to the most recent training.
                                                                       |

    Methods 
    -------
    sign(cls, input_value) [class method]
        Used as an activation function.

    linear_1D(cls, input_value) [class method]
        Returns 'input_value'. Used as an activation function.

    perceptron_loss_function(cls, predicted_outputs, target_outputs)
            [class method]
        The loss function for the perceptron algorithm.

    perceptron_stochastic_gradient(cls, predicted_output, target_output)
            [class method]
        The gradient of the perceptron's loss function when it employs
        stochastic descent.

    linear_regression_loss_function(cls, predicted_outputs, 
            target_outputs) [class method]
        The loss function for the linear regression algorithm.

    linear_regression_1D_stochastic_gradient(cls, predicted_output, 
            target_output) [class method]
        The gradient of the one-dimensional linear regression loss 
        function when it employs stochastic descent.

    preactivation(cls, input, weights, bias) [class method]
        The preactivation function for the single neuron (dot product of
        weights and input vector, plus bias)

    __init__(self, data_dimension, model_type, weights=None, bias=None, 
            activation_function=None)
        Initializes a model for a given dimensionality of training data
        and model type. Can "warm start" with a user-provided weight and
        bias, otherwise randomizes them. Can also provide an activation
        function if the user wants one different from the default for 
        the model type.

<<<<<<< HEAD
    predict(self, inputs, weights=None, bias=None, 
=======
    predict_outputs(self, inputs, weights=None, bias=None, 
>>>>>>> fbf9b83d24e0cfa2e2d248d8b2ad2d9a9247d88e
            use_current_weights_and_bias=True)
        Outputs the model's predictions given an array of feature 
        vectors. 

    current_weights_and_bias(self)
        Utility function for wrapping weights and bias in a single tuple.

    perceptron_stochastic_gradient_update(self, input, target_output)
        Updates the weights of the perceptron model using stochastic
        descent.
    
    linear_regression_1D_stochastic_gradient_update(self, input, 
            target_output, learning_rate, training_data_length)
        Updates the weights of the linear regression model using 
        stochastic descent.
    
    train(self, inputs, target_outputs, learning_rate=0.5, 
            num_epochs=50)
        Trains the model (updtes weights) using a batch of inputs and 
        target outputs.

    reset_model(self)
        Randomizes the weights and bias of the model.
    
    forget_previous_training(self)
        Resets the weights and bias to their values before the most 
        recent training.
                                                                       |
    """
    type_perceptron = "perceptron"
    type_linear_regression_1D = "linear regression 1D"

    @classmethod
    def sign(cls, input_value):
        """ 
        Returns the sign of input_value (+1 if it's positive, -1 if
        it's negative). 
        """
        if input_value >= 0:
            return 1
        else: 
            return -1
        
    @classmethod
    def linear_1D(cls, input_value):
        """ 
        Returns input_value. 
        """
        return input_value

    @classmethod
    def perceptron_loss_function(cls, 
                                 predicted_outputs, 
                                 target_outputs):
        """ 
        The perceptron loss function given the perceptron's predictions
        and the target outputs.

        Parameters
        ----------
        predicted_outputs : array_like
            Perceptron's predictions. 

        target_outputs : array_like
            Targets we`re training the perceptron to meet.

        Returns
        -------
        loss : float
            The error of the prediction.
        """
        return (1/4) * np.sum((predicted_outputs - target_outputs)**2)
    
    @classmethod
    def perceptron_stochastic_gradient(cls, 
                                       predicted_output, 
                                       target_output):
        """ 
        The gradient of the perceptron's loss function when it employs
        stochastic descent.

        Parameters
        ----------
        predicted_output : array_like
            Perceptron's predictions. 

        target_output : array_like
            Targets we`re training the perceptron to meet.

        Returns
        -------
        gradient : float
            The gradient of the loss function at the weights used to 
            calculate predicted_output
        """
        return (1/2) * (predicted_output - target_output)

    @classmethod
    def linear_regression_loss_function(cls, 
                                        predicted_outputs, 
                                        target_outputs):
        """ 
        The loss function for the linear regression algorithm.

        Parameters
        ----------
        predicted_outputs : array_like
            Perceptron's predictions. 

        target_outputs : array_like
            Targets we`re training the perceptron to meet.

        Returns
        -------
        loss : float
            The error of the prediction.
        """
        return (1/(2*target_outputs.size)) \
                * np.sum((predicted_outputs - target_outputs)**2)
        
    @classmethod
    def linear_regression_1D_stochastic_gradient(cls, 
                                                 predicted_output, 
                                                 target_output, 
                                                 training_data_length):
        """ 
        The gradient of the one-dimensional linear regression loss 
        function when it employs stochastic descent.

        Parameters
        ----------
        predicted_output : array_like
            The model's predictions. 

        target_output : array_like
            Targets we`re training the model to meet.

        Returns
        -------
        gradient : float
            The gradient of the loss function at the weights used to 
            calculate predicted_output

        """
        return (1/training_data_length) * (predicted_output - target_output)

    @classmethod
    def preactivation(cls, input, weights, bias):
        """ 
        The preactivation function for the single neuron (dot product of
        weights and input vector, plus bias)
        
        Parameters
        ----------
        input : array_like
            The vector input to the neuron.
        
        weights : array_like
            The vector of weights of the model.
        
        bias : float
            The bias of the model.
        
        Returns
        -------
        preactivation_value
            The dot product of input and weights, plus the bias.
        """
        if not (np.shape(input) == np.shape(weights)):
            raise ValueError("Input vector must have the same shape as weights"
                + "vector." + f"{np.shape(input) = },  {np.shape(weights) = }")
        
        return np.dot(input, weights) + bias
    
    def __init__(self, 
                 data_dimension, 
                 model_type, 
                 weights=None, 
                 bias=None, 
                 activation_function=None):
        """
        Initializes a model for a given dimensionality of training data
        and model type.

        Parameters
        ----------
        data_dimension : int
            The dimensionality of the feature vectors. Shouldn't be modified
            after initialization.

        model_type : str
            The category of model. Used to select the activation function and 
            loss function. Shouldn't be modified post-initialization.

        weights : array_like
            Optional; "warm start" weights can be provided by the user, 
            otherwise weights are generated from a normal distribution.

        bias : array_like
            Optional; "warm start" bias can be provided by the user, 
            otherwise bias is generated from a normal distribution.
        
        activation_function : function
            Optional; an arbitrary activation function can be provided by the 
            user, otherwise the standard function for the model type is 
            selected.
        """
        if data_dimension < 1:
            raise ValueError(f"Provided data dimension is {data_dimension} " 
                             + "but it must be a positive integer.")
        self.data_dimension = data_dimension 
        # A single number for each feature vector has dimension 1, 
        # a 2D vector has dimension 2, etc.

        self.model_type = model_type

        if activation_function is None:
            if self.model_type == SingleNeuron.type_perceptron:
                self.activation_function = SingleNeuron.sign
            elif self.model_type == SingleNeuron.type_linear_regression_1D:
                self.activation_function = SingleNeuron.linear_1D
        else:
            self.activation_function = activation_function
        
        if weights is None:
            self.weights = np.random.randn(data_dimension)
            if data_dimension == 1:
                # Unwrap weights to a scalar if there's only one weight
                # This avoids additional checks in preactivation
                self.weights = self.weights[0]
        else:
            self.weights = weights
        self.previous_weights = self.weights

        if bias is None:
            self.bias = np.random.randn()
        else:
            self.bias = bias
        self.previous_bias = self.bias

    def predict(self, 
                        inputs, 
                        use_current_weights_and_bias=True,
                        weights=None, 
                        bias=None):
        """ 
        Outputs the model's predictions given an array of feature 
        vectors. 

        Parameters
        ----------
        inputs : array_like
            An array of feature vectors for the model to predict the outputs
            of. Array dimensions incompatible with the dimensionality of the 
            model will raise a ValueError. Compatible array shapes are:
                - Equal shape to that of the weights vector (single input)
                - Final dimension equal to that of the weights vector (multiple
                inputs)

        use_current_weights_and_bias : boolean
            Optional; if desired, the user can set this to false and provide 
            their own weights and bias whenever they require a prediction. By 
            default set to true, so this function uses the model's current 
            weights and bias.

        weights : array_like
            Optional; weight vector to use if the user desires to calculate 
            the output with weights other than the model's current weights.
        
        bias : float
            Optional; bias to use if the user desires to calculate 
            the output with bias other than the model's current bias.

        Returns
        -------
        outputs : array_like
            The model's outputs corresponding to each input.
        """
        if use_current_weights_and_bias:
            weights = self.weights
            bias = self.bias

        if np.isscalar(inputs):
            if self.data_dimension != 1:
                raise ValueError("Mismatch between expected feature vector "
                    + f"dimension ({self.data_dimension = }) and input " 
                    + f"shape ({np.shape(inputs) = }).")
            else:
                return self.activation_function(
                    SingleNeuron.preactivation(inputs, weights, bias))

        elif inputs.shape == (self.data_dimension,):
            return self.activation_function(
                SingleNeuron.preactivation(inputs, weights, bias))
        
        elif inputs.shape[-1] == self.data_dimension    \
                or (self.data_dimension == 1 and inputs.ndim == 1):
            return [self.activation_function(
                        SingleNeuron.preactivation(input, weights, bias)) 
                    for input in inputs]

        else: 
            raise ValueError("Mismatch between expected feature vector "
                + f"dimension ({self.data_dimension = }) and input " 
                + f"shape ({np.shape(inputs) = }).")
    
    def current_weights_and_bias(self):
        """ 
        Utility function for wrapping weights and bias in a single tuple.
        """
        return (self.weights.copy(), self.bias)
    
    def perceptron_stochastic_gradient_update(self, 
                                              input, 
                                              target_output):
        """ 
        Updates the weights of the perceptron model using stochastic
        descent. 
        
        Parameters
        ----------
        input : array_like
            The single input used to calculate the stochastic gradient at the
            current weights.
        
        target_output : array_like
            The single target output used to calculate the stochastic gradient
            at the current weights.
        
        Returns
        -------
        gradient : float
            The calculated gradient used to update the weights. 

        """
        gradient = SingleNeuron.perceptron_stochastic_gradient(
                        self.predict(input), target_output)
        self.weights -= gradient * input
        self.bias -= gradient
        return gradient

    def linear_regression_1D_stochastic_gradient_update(self, 
                                                        input, 
                                                        target_output, 
                                                        learning_rate, 
                                                        training_data_length):
        """ 
        Updates the weights of the linear regression model using 
        stochastic descent.
        
        Parameters
        ----------
        input : array_like
            The single input used to calculate the stochastic gradient at the
            current weights.
        
        target_output : array_like
            The single target output used to calculate the stochastic gradient
            at the current weights.
        
        Returns
        -------
        gradient : float
            The calculated gradient used to update the weights.

        """
        gradient = SingleNeuron.linear_regression_1D_stochastic_gradient(
                                                self.predict(input), 
                                                target_output, 
                                                training_data_length)
        self.weights -= learning_rate * gradient * input
        self.bias -= learning_rate * gradient
        return gradient
        
    def train(self, 
              inputs, 
              target_outputs, 
              learning_rate=0.5, 
              num_epochs=50):
        """ 
        Trains the model (updates weights) using a batch of inputs and 
        target outputs.
        
        Parameters
        ----------
        inputs : array_like
            An array of feature vectors to train the model on. Must be as 
            many feature vectors as there are target outputs.
        
        target_outputs : array_like
            An array of target outputs to compare the model's predictions to. 
            Must be as many targets as there are feature vectors.

        learning_rate : float
            Optional, and will not be used for the perceptron model; defines 
            the distance along the gradient by which the weights are adjusted 
            at every iteration.

        num_epochs : int
            Optional; defines the number of times the model looks at every 
            input-target pair. I.e. in each epoch the model looks at every
            pair.
        
        Returns
        -------
        loss_at_epoch : numpy.ndarray
            An array of the value of the model's loss function at each epoch
            in this training.
        
        """
        self.previous_weights = np.copy(self.weights)
        self.previous_bias = np.copy(self.bias)
        weight_update = None
        loss_function = None
        
        if self.model_type == SingleNeuron.type_perceptron:
            weight_update = self.perceptron_stochastic_gradient_update
            loss_function = SingleNeuron.perceptron_loss_function
            
        elif self.model_type == SingleNeuron.type_linear_regression_1D:
            weight_update = lambda input, target : \
                self.linear_regression_1D_stochastic_gradient_update(
                        input, target, learning_rate, len(target_outputs))
            loss_function = SingleNeuron.linear_regression_loss_function

        loss_at_epoch = np.empty(1 + num_epochs)
        loss_at_epoch[0] = loss_function(self.predict(inputs),
                                         target_outputs)

        for epoch_index in ShadyBar(
                "Training", 
                suffix="Epoch %(index)d / %(max)d").iter(range(num_epochs)):
            
            for input, target_output in zip(inputs, target_outputs):
                weight_update(input, target_output)
                
            loss_at_epoch[epoch_index+1] = loss_function(
                    self.predict(inputs), target_outputs)

        self.prev_loss_history = self.loss_history.copy()
        self.loss_history.extend(loss_at_epoch)

        return loss_at_epoch

    def reset_model(self):
        """ 
        Randomizes the weights and bias of the model.
        """
        if self.data_dimension == 1:
            self.weights = np.random.randn()
        else: 
            self.weights = np.random.randn(self.weights.shape)
        self.bias = np.random.randn()

        self.loss_history = []
        self.prev_loss_history = []

    def forget_previous_training(self):
        """ 
        Resets the weights and bias to their values before the most recent
        training.
        """
        self.weights = np.copy(self.previous_weights)
        self.bias = np.copy(self.previous_bias)
        self.loss_history = self.prev_loss_history.copy()

    def __repr__(self):
        return "Single neuron model type: " + self.model_type \
            + " || Weights: " + self.weights + " | Bias: " + self.bias

    def plot_loss_history(self, plt_figure=None, loss_label=None):
        
        if loss_label is None:
            loss_label = "Loss in " + self.__repr__()

        loss_plot = None

        if plt_figure is not None:
            loss_plot = plt.plot(range(1, len(self.loss_history)+1), 
                                 self.loss_history, 
                                 label=loss_label, 
                                 figure=plt_figure)
        else:
            plt.figure(figsize=(10,8))
            loss_plot = plt.plot(range(1, len(self.loss_history)+1), 
                                 self.loss_history, 
                                 label=loss_label)
            plt.legend(fontsize=15)
            plt.xlabel("Epoch number", fontsize=15)
            plt.ylabel("Loss value", fontsize=15)
            plt.title("Loss over epochs", fontsize=18)
            plt.show()

        return loss_plot

        