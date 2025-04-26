module SingleNeuronJulia

using LinearAlgebra, DataFrames, Plots

export SingleNeuron, predict, train!,
        forgetprevtraining!,
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

const type_perceptron = :perceptron
const type_linearregression = :linearregression
const type_logisticregression = :logisticregression
    
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

function predict(neuron::SingleNeuron, input)
    return neuron.activationfunction(preactivation(neuron, input))
end

function predict(neuron::SingleNeuron, inputs::DataFrame)
    return [predict(neuron, input) for input in eachrow(inputs)]
end

function predict(neuron::SingleNeuron, inputs::Vector{<:Vector})
    return [predict(neuron, input) for input in inputs]
end

function predict(neuron::SingleNeuron, inputs::AbstractRange)
    return [predict(neuron, input) for input in inputs]
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

function updateweightsingle!(neur::SingleNeuron, input, target, 
                       learningrate)
    gradient = neur.gradient(predict(neur, input), target)
    neur.weights .-= (learningrate * gradient) .* input
    neur.bias -= learningrate * gradient
end

function updateweightsmultiple!(neur::SingleNeuron, inputs, targets, 
                                learningrate)
    for (input, target) in zip(inputs, targets)
        gradient = neur.gradient(predict(neur, input), target)
        neur.weights .-= (learningrate * gradient) .* input
        neur.bias -= learningrate * gradient
    end
end

# Generally, assumes that you're putting in multiple input-target pairs...
function updateweights!(neur::SingleNeuron, inputs, targets, learningrate)
    return updateweightsmultiple!(neur, inputs, targets, learningrate)
end

# ...but we have to do a check if we get a vector of numbers, just in case 
# the feature vectors are one-dimensional and the vector of numbers isn't a 
# feature vector but is instead a vector of feature scalars
function updateweights!(neur::SingleNeuron, inputs::Vector{<:Number}, targets, 
                       learningrate)
    if length(neur.weights) == 1
        return updateweightsmultiple!(neur, inputs, targets, learningrate)
    else
        return updateweightsingle!(neur, inputs, targets, learningrate)
    end
end

function updateweights!(neur::SingleNeuron, inputs::DataFrame, targets, 
                       learningrate)
    return updateweights!(neur, Vector.(eachrow(inputs)), targets, learningrate)
end

function trainloop!(neur::SingleNeuron, inputs, targets, 
                    numepochs, learningrate; 
                    lossatepoch = zeros(numepochs+1), 
                    weightupdate! = updateweightsmultiple!)
        
    lossatepoch[begin] = neur.loss(predict(neur, inputs), targets)

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
        
        lossatepoch[epoch+1] = neur.loss(predict(neur, inputs), targets)
    end

    return lossatepoch
end

function trainloop!(neur::SingleNeuron, inputs::Vector{<:Number}, targets, 
                    numepochs, learningrate; 
                    lossatepoch = zeros(numepochs+1))
    if length(neur.weights) == length(inputs)
        return trainloop!(neur, inputs, targets, numepochs, learningrate; 
                          lossatepoch=lossatepoch, 
                          weightupdate! = updateweightsingle!)
    else
        return trainloop!(neur, inputs, targets, numepochs, learningrate; 
                          lossatepoch=lossatepoch, 
                          weightupdate! = updateweightsmultiple!)
    end
end

function trainloop!(neur::SingleNeuron, inputs::DataFrame, targets, 
                    numepochs, learningrate; 
                    lossatepoch=zeros(numepochs+1))
    return trainloop!(neur, Vector.(eachrow(inputs)), targets, numepochs, 
                      learningrate; lossatepoch=lossatepoch)
end

function checkdatalengths(inputs, targets)
    return length(inputs) == length(targets)
end

function checkdatalengths(inputs::DataFrame, targets)
    return nrow(inputs) == length(targets)
end

function train!(neur::SingleNeuron, inputs, targets; 
                numepochs=50, learningrate=0.005)
    if !checkdatalengths(inputs, targets)
        error("Input and target arrays must be of the same length")
    end

    if isequal(neur.gradient, perceptronstochasticgradient)
        learningrate = 1
    end

    copy!(neur.previousweights, neur.weights)
    neur.previousbias = neur.bias

    lossatepoch = trainloop!(neur, inputs, targets, numepochs, 
                             learningrate)

    neur.prevlosshistory = neur.losshistory
    neur.losshistory = [neur.losshistory; lossatepoch]
    return lossatepoch
end

function train!(neur::SingleNeuron, inputs::DataFrame, targets;
                numepochs=50, learningrate = 0.005)
    return train!(neur, Vector.(eachrow(inputs)), targets; 
                  numepochs=numepochs, learningrate=learningrate)
end

function forgetprevtraining!(neur::SingleNeuron)
    copy!(neur.weights, neur.previousweights)
    neur.bias = neur.previousbias
    copy!(neur.losshistory, neur.prevlosshistory)
end

function linerepresentation(neur::SingleNeuron, x)
    if length(neur.weights) != 2
        error("This neuron has $(length(neur.weights)) weights, \
               so can't be represented as a line.")
    else
        return (neur.weights[1])
    end
end

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

function plotneuron!(neur::SingleNeuron; leftbound=0, rightbound=1)
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

end # SingleNeuronJulia module

using .SingleNeuronJulia