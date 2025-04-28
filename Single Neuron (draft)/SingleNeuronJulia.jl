module SingleNeuronJulia

using LinearAlgebra, DataFrames, Plots

export SingleNeuron, predict, train!,
        forgetprevtraining!, plotneuron, 
        plotneuron!, plotlosshistory, 
        plotlosshistory!, linerepresentation,
        sign_zeropositive, linear, sigmoid, 
        perceptronloss, linearregressionloss, 
        binarycrossentropyloss, meansquarederror, 
        perceptronstochasticgradient, 
        regressionstochasticgradient

"""
A struct representing a single machine learning neuron. 
Defined by an activation function, gradient function, 
and a loss function. Its weights are mutable. The types 
are all concrete to ensure efficiency.

Type parameters
---------------
ActivatorF: the type of the activation function.

GradientF: the type of the gradient function.

LossF: the type of the loss function.

Fields
------
activationfunction::ActivatorF
    The neuron's immutable activation function. In `predict` 
    it's assumed to take a single argument. 

gradient::GradientF
    The neuron's immutable gradient function. In `predict` 
    it's assumed to take two arguments.

loss::LossF
    The neuron's immutable loss function. In `predict` it's 
    assumed to take two arguments.

weights::Vector{Float64}
    The neuron's current weights. `length(weights)` defines 
    how many elements the neuron's inputs must have.

bias::Float64
    The neuron's current bias.

previousweights::Vector{Float64}
    The neuron's weights previous to the most recent 
    training.
    
previousbias::Float64
    The neuron's bias previous to the most recent training.

losshistory::Array{Float64,1}
    The value of the loss function after each epoch of 
    training, given the inputs and targets from that training.

prevlosshistory::Array{Float64,1}
    The value of `losshistory` previous to the most recent
    training.
"""
mutable struct SingleNeuron{ActivatorF, GradientF, LossF}
    const activationfunction::ActivatorF
    const gradient::GradientF
    const loss::LossF
    weights::Vector{Float64}
    bias::Float64
    previousweights::Vector{Float64}
    previousbias::Float64
    losshistory::Array{Float64,1}
    prevlosshistory::Array{Float64,1}
end 

# Symbols used to delineate preset single-neuron models.
const type_perceptron = :perceptron
const type_linearregression = :linearregression
const type_logisticregression = :logisticregression

"""
Constructs a `SingleNeuron` with given defining functions and
initial weights and bias.
"""
function SingleNeuron(activationfunction::Function, 
                      gradientfunction::Function, 
                      lossfunction::Function, 
                      weights::Vector{Float64}, 
                      bias::Float64)
    return SingleNeuron{typeof(activationfunction), 
                        typeof(gradientfunction), 
                        typeof(lossfunction)}(activationfunction, 
                                              gradientfunction, 
                                              lossfunction, 
                                              weights, bias, weights, 
                                              bias, [], [])
end

"""
Constructs a `SingleNeuron` using predefined types.

Positional arguments
--------------------
datadimension::Int
    The number of elements input feature vectors must have. The 
    number of weights.

modeltype::Symbol
    One of the following symbols for predefined types.
    - `:perceptron` defines a perceptron employing stochastic 
    gradient descent.
    - `:linearregression` defines a one-dimensional linear 
    regression model employing stochastic gradient descent.
    Multidimensional linear regression is uh... currently undefined
    (might work but I'm not sure)
    - `:logisticregression` defines a logistic regression classifier
    employing stochastic gradient descent.

Keyword arguments
-----------------
weights::Vector{Float64} = zeros(Float64, datadimension)
    Initial weights. Providing a vector of length other than 
    `datadimension` will throw an error.

bias::Float64 = 0.0
    Initial bias.
"""
function SingleNeuron(datadimension::Int, modeltype::Symbol; 
            weights::Vector{Float64}=zeros(Float64, datadimension), 
            bias::Float64=0.0)
    if length(weights) != datadimension
        error("Provided weight vector has length $(length(weights)) \
                not of the supplied dimension $datadimension")
    end

    if modeltype == type_perceptron
        return SingleNeuron(sign_zeropositive, perceptronstochasticgradient, 
                            perceptronloss, weights, bias)
    elseif modeltype == type_linearregression
        return SingleNeuron(linear, regressionstochasticgradient, 
                            linearregressionloss, weights, bias)
    elseif modeltype == type_logisticregression
        return SingleNeuron(sigmoid, regressionstochasticgradient, 
                            binarycrossentropyloss, weights, bias)
    else
        error("\"$modeltype\" not a recognized model identifier.")
    end
end

"""
Preactivation function given `neuron`'s weights. Uses `dot` rather than 
    broadcasting so that it throws an error if the dimensions of 
    `input` and `neuron.weights` are not equal.
"""
preactivation(neuron::SingleNeuron, input) = (dot(input, neuron.weights) 
                                              + neuron.bias)

"""
Preactivation function. Uses `dot` rather than broadcasting so that 
    it throws an error if the dimensions of `input` and 
    `neuron.weights` are not equal.
"""
preactivation(input, weights, bias) = dot(input, weights) + bias

"""
Uses `neuron`'s weights to predict the output given a single 
input. Not an exported function.
"""
function predictsingle(neuron::SingleNeuron, input)
    return neuron.activationfunction(preactivation(neuron, input))
end

"""
Uses `neuron`'s weights to predict the output given multiple inputs.
Not an exported function.
"""
function predictmultiple(neuron::SingleNeuron, inputs)
    return [predictsingle(neuron, input) for input in inputs]
end

"""
Uses `neuron`'s weights to predict the output given a single 
input.
"""
function predict(neuron::SingleNeuron, input)
    return neuron.activationfunction(preactivation(neuron, input))
end

"""
Uses `neuron`'s weights to predict the output given a vector 
input. Whether `input` is treated as a single input or multiple
inputs depends on whether the dimension of `input` matches that 
of `neuron.weights`.
"""
function predict(neuron::SingleNeuron, input::Vector{<:Number})
    if length(neuron.weights) == length(input)
        return predictsingle(neuron, input)
    else
        return predictmultiple(neuron, input)
    end
end

"""
Uses `neuron`'s weights to predict the output given a `DataFrame`
input.
"""
function predict(neuron::SingleNeuron, inputs::DataFrame)
    return [predictsingle(neuron, input) 
            for input in eachrow(inputs)]
end

"""
Uses `neuron`'s weights to predict the output given multiple inputs.
Treats each vector in `inputs` as a single input.
"""
function predict(neuron::SingleNeuron, inputs::Vector{<:Vector})
    return predictmultiple(neuron, inputs)
end

"""
Uses `neuron`'s weights to predict the output given a range of 
inputs. Treats each element of `inputs` as a single input.
"""
function predict(neuron::SingleNeuron, inputs::AbstractRange)
    return predictmultiple(neuron, inputs)
end

"Returns -1 if the argument is less than 0, 1 otherwise."
sign_zeropositive(value) = value < 0.0 ? -1 : 1

"Returns the argument."
linear(value) = value

"""
Returns the value of the sigmoid function at input_value.
    
    https://en.wikipedia.org/wiki/Sigmoid_function 
"""
sigmoid(value) = 1.0 ./ (1.0 .+ exp.(-value))

"The perceptron's loss function."
perceptronloss(predictions, targets) = 0.25 * sum((predictions .- targets).^2)

"The linear regression model's loss function."
function linearregressionloss(predictions, targets)
    return ( (1 / 2(length(targets))) * sum((predictions .- targets).^2) )
end

"""The binary cross entropy loss function. Used for logistic regression 
classification."""
function binarycrossentropyloss(predictions, targets)
    return ( (1 / length(targets)) 
            * sum(-targets .* log.(predictions)
                  - ((1 .- targets).*log.(1 .- predictions))) )
end

"""Mean squared error between `predictions` and `targets`. Used as a 
loss function."""
meansquarederror(predictions, targets) = 0.5 .* (predictions .- targets).^2

"The gradient function used in the perceptron model."
perceptronstochasticgradient(prediction, target) = 0.5 * (prediction .- target)

"The gradient function used in the linear and logistic regression models."
regressionstochasticgradient(prediction, target) = prediction .- target

"""Updates the weights of `neur` given a single input, corresponding 
target, and a learning rate. Not an exported function."""
function updateweightsingle!(neur::SingleNeuron, input, target, 
                       learningrate)
    gradient = neur.gradient(predictsingle(neur, input), target)
    neur.weights .-= (learningrate * gradient) .* input
    neur.bias -= learningrate * gradient
end

"""Updates the weights of `neur` given multiple inputs, corresponding 
targets, and a learning rate. Not an exported function."""
function updateweightsmultiple!(neur::SingleNeuron, inputs, targets, 
                                learningrate)
    for (input, target) in zip(inputs, targets)
        gradient = neur.gradient(predictsingle(neur, input), target)
        neur.weights .-= (learningrate * gradient) .* input
        neur.bias -= learningrate * gradient
    end
end

# Generally, assumes that you're putting in multiple input-target pairs...
"""Updates the weights of `neur` given multiple inputs, corresponding 
targets, and a learning rate."""
function updateweights!(neur::SingleNeuron, inputs, targets, learningrate)
    return updateweightsmultiple!(neur, inputs, targets, learningrate)
end

# ...but we have to do a check if we get a vector of numbers, just in case 
# the feature vectors are one-dimensional and the vector of numbers isn't a 
# feature vector but is instead a vector of feature scalars
"""
Updates the weights of `neur` given input vector(s), corresponding target(s), 
and a learning rate. Whether `inputs` is treated as a single input 
or multiple inputs depends on whether the dimension of `neuron.weights` is 1.
"""
function updateweights!(neur::SingleNeuron, inputs::Vector{<:Number}, targets, 
                       learningrate)
    if length(neur.weights) == 1
        return updateweightsmultiple!(neur, inputs, targets, learningrate)
    else
        return updateweightsingle!(neur, inputs, targets, learningrate)
    end
end

"""Updates the weights of `neur` given a `DataFrame` of input(s), 
corresponding target(s), and a learning rate."""
function updateweights!(neur::SingleNeuron, inputs::DataFrame, targets, 
                        learningrate)
    return updateweights!(neur, Vector.(eachrow(inputs)), targets, learningrate)
end

"""
Trains 'neur' on 'inputs' and 'targets' over 'numepochs' epochs at 
a rate of 'learningrate'. 
Returns the value of the loss function at each epoch of training. 
Not an exported function. 

Optional: supply a boolean value 'ismultipleinputs', depending on 
whether you want to train on multiple input/target pairs or just a single 
one. Helps with efficiency.
"""
function trainloop!(neur::SingleNeuron, inputs, targets, 
                    numepochs, learningrate; 
                    lossatepoch = zeros(numepochs+1), 
                    ismultipleinputs = true)
    
    if ismultipleinputs 
        weightupdate! = updateweightsmultiple!
        predictfunc = predictmultiple
    else 
        weightupdate! = updateweightsingle!
        predictfunc = predictsingle
    end

    lossatepoch[begin] = neur.loss(predictfunc(neur, inputs), targets)

    tempweights=copy(neur.weights)
    for epoch in 1:numepochs
        
        copy!(tempweights, neur.weights)
        tempbias = neur.bias

        weightupdate!(neur, inputs, targets, learningrate)

        if (any(isinf.(neur.weights)) || any(isnan.(neur.weights)) 
                || isinf(neur.bias) || isnan(neur.bias))
            copy!(neur.weights, tempweights)
            neur.bias = tempbias
            error("Model has diverged. Try turning down the learning rate.\n\
                Previous weights: $(tempweights) | \
                Previous bias: $(tempbias) | Epoch: $(epoch)")
        end
        
        lossatepoch[epoch+1] = neur.loss(predictfunc(neur, inputs), targets)
    end

    return lossatepoch
end

# General behavior assumes `inputs` is an iterator of multiple inputs
"""Runs `trainloop!` assuming we're training on multiple inputs. Dispatch is 
faster than checking type. 
Returns the value of the loss function at each epoch of training.
Not an exported function."""
function dispatchtraining!(neur::SingleNeuron, inputs, targets, 
                           numepochs, learningrate; 
                           lossatepoch=zeros(numepochs+1))
    return trainloop!(neur, inputs, targets, numepochs, 
                      learningrate; lossatepoch=lossatepoch)
end

"""Runs `trainloop!` assuming either multiple inputs or a single input 
depending on whether the length of `neur.weights` matches that of `inputs`. 
Dispatch is faster than checking type.
Returns the value of the loss function at each epoch of training. 
Not an exported function."""
function dispatchtraining!(neur::SingleNeuron, inputs::Vector{<:Number}, 
                           targets, numepochs, learningrate; 
                           lossatepoch = zeros(numepochs+1))
    if length(neur.weights) == length(inputs)
        return trainloop!(neur, inputs, targets, numepochs, learningrate; 
                          lossatepoch=lossatepoch, ismultipleinputs=false)
    else
        return trainloop!(neur, inputs, targets, numepochs, learningrate; 
                          lossatepoch=lossatepoch)
    end
end

"""Runs `trainloop!` assuming we're training on a DataFrame, which may be 
composed of a single row or multiple. Dispatch is 
faster than checking type. Not an exported function.
Returns the value of the loss function at each epoch of training."""
function dispatchtraining!(neur::SingleNeuron, inputs::DataFrame, targets, 
                    numepochs, learningrate; 
                    lossatepoch=zeros(numepochs+1))
    return dispatchtraining!(neur, Vector.(eachrow(inputs)), targets, numepochs, 
                      learningrate; lossatepoch=lossatepoch)
end

"""Returns whether `inputs` and `targets` have the same length for training 
purposes. Not an exported function."""
function equaldatalengths(inputs, targets)
    return length(inputs) == length(targets)
end

"""Returns whether `inputs` and `targets` have the same length for training 
purposes. Not an exported function."""
function equaldatalengths(inputs::DataFrame, targets)
    return nrow(inputs) == length(targets)
end

"""
Employ `neur`'s activation function, gradient function, and loss function 
to train `neur` on `inputs` and `targets`. Stores values in 
`previousweights` and `previousbias` before training, and stores values 
in `prevlosshistory` after training.

Positional arguments
--------------------
neur
    The `SingleNeuron` to be trained.

inputs
    The inputs on which to train `neur`. May be a collection or a scalar.
    Number of inputs must match number of targets.

targets 
    The targets on which to train `neur`. May be a collection or a scalar.
    Number of inputs must match number of targets.

Keyword arguments
-----------------
numepochs = 50
    The number of epochs over which to train `neur`.

learningrate = 0.005
    The learning rate for this training. If `neur` is a perceptron then 
    this learning rate is discarded, because the perceptron trains at a 
    universal rate.

Returns
-------
The value of the loss function at each epoch of training.
"""
function train!(neur::SingleNeuron, inputs, targets; 
                numepochs=50, learningrate=0.005)
    if !equaldatalengths(inputs, targets)
        error("Input and target arrays must be of the same length")
    end

    if isequal(neur.gradient, perceptronstochasticgradient)
        learningrate = 1
    end

    copy!(neur.previousweights, neur.weights)
    neur.previousbias = neur.bias

    lossatepoch = dispatchtraining!(neur, inputs, targets, numepochs, 
                                    learningrate)

    neur.prevlosshistory = neur.losshistory
    neur.losshistory = [neur.losshistory; lossatepoch]
    return lossatepoch
end

"""
Run `train!` on a `DataFrame` of inputs by converting `inputs` to 
a vector of rows.
"""
function train!(neur::SingleNeuron, inputs::DataFrame, targets;
                numepochs=50, learningrate = 0.005)
    return train!(neur, Vector.(eachrow(inputs)), targets; 
                  numepochs=numepochs, learningrate=learningrate)
end

"""Discard the current weights, bias, and loss history of 'neur', 
replacing them with `neur.previousweights`, `neur.previousbias`, 
and `neur.prevlosshistory`."""
function forgetprevtraining!(neur::SingleNeuron)
    copy!(neur.weights, neur.previousweights)
    neur.bias = neur.previousbias
    copy!(neur.losshistory, neur.prevlosshistory)
end

"""
Function representing the linear representation of a 
two-dimensional neuron like a perceptron or logistic regression 
classifier.
"""
function linerepresentation(neur::SingleNeuron, x)
    if length(neur.weights) != 2
        error("This neuron has $(length(neur.weights)) weights, \
               so can't be represented as a line.")
    else
        return (neur.weights[1] .* x .+ neur.bias) / -neur.weights[2]
    end
end

"""
Plots a representation of the weights of a neuron on a new plot. 
If the neuron is two-dimensional it plots the line defined by the 
weights and bias. If the neuron is one-dimensional it plots the outputs 
of the neuron given an input range bounded by `leftbound` and 
`rightbound`. Passes `kw` directly to the `Plots.plot` function.
"""
function plotneuron(neur::SingleNeuron; leftbound=0, rightbound=1,
                    kw...)
    if length(neur.weights) == 2
        plotdomain = range(leftbound, rightbound, 100)
        plot(plotdomain, linerepresentation(neur, plotdomain); kw...)
    elseif length(neur.weights) == 1
        plotdomain = range(leftbound, rightbound, 100)
        plot(plotdomain, predict(neur, plotdomain); kw...)
    else
        error("Neuron has dimension $(length(neur.weights)) so can't \
               be plotted in 2D.")
    end
end

"""
Plots a representation of the weights of a neuron on an existing plot. 
If the neuron is two-dimensional it plots the line defined by the 
weights and bias. If the neuron is one-dimensional it plots the outputs 
of the neuron given an input range bounded by `leftbound` and 
`rightbound`. Passes `kw` directly to the `Plots.plot` function.
"""
function plotneuron!(neur::SingleNeuron; leftbound=0, rightbound=1, 
                     kw...)
    if length(neur.weights) == 2
        plotdomain = range(leftbound, rightbound, 100)
        plot!(plotdomain, linerepresentation(neur, plotdomain); kw...)
    elseif length(neur.weights) == 1
        plotdomain = range(leftbound, rightbound, 100)
        plot!(plotdomain, predict(neur, plotdomain); kw...)
    else
        error("Neuron has dimension $(length(neur.weights)) so can't \
               be plotted in 2D.")
    end
end

"""Plots the loss at each epoch over every training of this neuron 
on a new plot. Domain is the epoch count. Passes `kw` directly to the 
`Plots.plot` function."""
function plotlosshistory(neur::SingleNeuron; kw...)
    return plot(neur.losshistory, kw...)
end

"""Plots the loss at each epoch over every training of this neuron 
on an existing plot. Domain is the epoch count. Passes `kw` directly to the 
`Plots.plot` function."""
function plotlosshistory!(neur::SingleNeuron; kw...)
    return plot!(neur.losshistory, kw...)
end

end # SingleNeuronJulia module

using .SingleNeuronJulia