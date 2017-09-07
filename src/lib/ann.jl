#A neural network in Julia


cons{T}(x::T, ys::Array{T, 1})::Array{T, 1} = prepend!(copy(ys), x)
    

function sigmoid(z::Float64)::Float64
    1.0 ./ (1.0 .+ exp(-z))
end

mutable struct Layer
    size::Int32
    Z::Array{Float64, 1}
    A::Array{Float64, 1}
    W::Array{Float64, 2}
    b::Array{Float64, 1}
    g::Function

    Layer(n::Int, nprev::Int, activation::Function) =
        new(n,
            zeros(n),
            zeros(n),
            0.01 * rand(Float64, n, nprev),
            zeros(n),
            activation)
end

function forward_propagate(input::Layer, output::Layer)
    output.Z = output.W * input.A + output.b
    output.activations = activate(output.logits)
end

mutable struct ANN
    depth::Int
    inputSize::Int
    layers::Array{Layer, 1}
    output::Float64

    ANN(ipsize::Int, hiddenLayerSize::Array{Int, 1}) =
        new(first(size(hiddenLayerSize)),
            ipsize,
            [Layer(n, nprev, sigmoid)
             for (n,nprev)=zip(hiddenLayerSize,
                               cons(ipsize, hiddenLayerSize)[1:end-1]) ],
             
            0.0)
end


    
