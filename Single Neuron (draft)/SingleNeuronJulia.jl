module SingleNeuronJulia

using LinearAlgebra

export SingleNeuron, predict, train,
        sign_zeropositive, linear, sigmoid, 
        perceptronloss, linearregressionloss, 
        binarycrossentropyloss, meansquarederror, 
        perceptronstochasticgradient, 
        regressionstochasticgradient

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

const type_perceptron = "perceptron"
const type_linearregression = "linear regression"
const type_logisticregression = "logistic regression"
    
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

function SingleNeuron(datadimension::Int; modeltype::String, 
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

### Actually... there's not a good reason to manually change the weights.
# Ignore this idea of changing setproperty!, since it involves weird type stuff
# function Base.setproperty!(x::SingleNeuron, s::Symbol, v)
#     if s == :weights
#         if length(v) != length(x.weights)
#             error("Tried to sets weights to array of length $(length(newval)), need length $(length(x.weights))")
#         else
#             setfield!(x, :weights, convert(Vector{Float64, v}))
#         end
#     end
# end

preactivation(neuron::SingleNeuron, input) = (dot(input, neuron.weights) 
                                              + neuron.bias)

preactivation(input, weights, bias) = dot(input, weights) + bias

function predict(neuron::SingleNeuron, inputs)
    return neuron.activationfunction.(preactivation(neuron, vec) 
                                      for vec in inputs)
end

"Returns -1 if the argument is less than 0, 1 otherwise."
sign_zeropositive(value) = value < 0.0 ? -1 : 1

"Returns the argument."
linear(value) = value

"""
Returns the value of the sigmoid function at input_value.
    
    https://en.wikipedia.org/wiki/Sigmoid_function 
"""
sigmoid(value) = 1.0 / (1.0 + exp(-value))

perceptronloss(predictions, targets) = 0.25 * sum((predictions .- targets).^2)

function linearregressionloss(predictions, targets)
    return ( (1 / 2(length(targets))) * sum((predictions .- targets).^2) )
end

function binarycrossentropyloss(predictions, targets)
    return ( (1 / length(targets)) 
            * sum(-targets .* log.(predictions)
                  - ((1 .- targets).*log.(1 .- predictions))) )
end

meansquarederror(predictions, targets) = 0.5 .* (predictions .- targets).^2

perceptronstochasticgradient(prediction, target) = 0.5 * (prediction .- target)

regressionstochasticgradient(prediction, target) = prediction .- target

function weight_update(neur::SingleNeuron, inputs, targets, numepochs; 
                       tempweights=(w = zeros(size(neur.weights)); copy!(w, neur.weights)), 
                       tempbias=neur.bias, lossatepoch=zeros(numepochs+1))
    lossatepoch[begin] = neur.loss(predict(neur, inputs))

    for epoch in 1:numepochs
        for (input, target) in zip(inputs, targets)
            gradient = neur.gradient(predict(neur, input), target)
            # Gradient is a scalar, since a single neuron can only output
            # a single value.
            tempweights .-= (learning_rate * gradient) .* input
            tempbias -= learning_rate * gradient
            if (any(isinf.(neur.weights)) || any(isnan.(neur.weights)) 
                    || isinf(neur.bias) || isnan(neur.bias))
                error("Model has diverged. Try turning down the learning rate.\n\
                        Previous weights: $(neur.weights) | \
                        Previous bias: $(neur.bias) | Epoch: $(epoch)")
            else
                copy!(neur.weights, tempweights)
                neur.bias = tempbias
            end
        end

        lossatepoch[epoch+1] = neur.loss(predict(neur, inputs), targets)
    end

    return lossatepoch
end

function train(neur::SingleNeuron, inputs, targets; 
        learning_rate=0.005, numepochs=50)
    if length(inputs) != length(targets)
        error("Input and target arrays must be of the same length")
    end

    if neur.gradient == perceptronstochasticgradient
        learning_rate = 1
    end

    copy!(neur.previousweights, neur.weights)
    neur.previousbias = neur.bias

    lossatepoch = weight_update(neur, inputs, targets, numepochs)

    neur.prevlosshistory = neur.losshistory
    return neur.losshistory = [neur.losshistory; lossatepoch]
end

function forgetprevtraining(neur::SingleNeuron)
    copy!(neur.weights, neur.previousweights)
    neur.bias = neur.previousbias
    copy!(neur.losshistory, neur.prevlosshistory)
end

end # SingleNeuronJulia module

using .SingleNeuronJulia