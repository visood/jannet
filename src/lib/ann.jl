#A neural network in Julia


cons{T}(x::T, ys::Array{T, 1})::Array{T, 1} = insert!(copy(ys), 1, x)#prepend!(copy(ys), [x])
cons{T}(ys::Array{T, 1}, x::T)::Array{T, 1} = push!(copy(ys), x)
    

sigmoid{T}(Z::T)::T = 1.0 ./ (1.0 .+ exp.(-Z))
sigmoidDeriv{T}(Z::T)::T = sigmoid(Z) .* (1.0 .- sigmoid(Z))

tanhDeriv{T}(Z::T)::T = 1.0 .- tanh.(Z) .^ 2

mutable struct Layer
    size::Int32                #number of neural units
    weights::Array{Float64, 2} #(size, size of previous layer)
    biases::Vector{Float64}    #(size, 1)
    activation::Function       #activation function

    Layer(n::Int, nprev::Int, activationFunction::Function) =
        new(n,
            0.01 * rand(Float64, n, nprev),
            zeros(n),
            activationFunction)
end

shape(l::Layer)::Tuple{Int, Int} = size(l.weights)

mutable struct ANN
    insize::Int           #input size
    depth::Int            #hidden layers + 1
    hidden::Vector{Layer} #hidden layers
    output::Layer         #output layer

    ANN(insize_::Int,
        hiddenSize::Vector{Int},
        outsize_::Int,
        hiddenActivation::Function,
        outputActivation::Function) =
        new(insize_,
            size(hiddenSize, 1) + 1,
            [ Layer(n, nprev, hiddenActivation)
              for(n, nprev)=zip(hiddenSize,
                                cons(insize_, hiddenSize)[1:end-1]) ],
            Layer(outsize_, hiddenSize[end], outputActivation) )

    ANN(insize_::Int,
        hiddenSize::Vector{Int},
        outsize_::Int) =
            ANN(insize_, hiddenSize, outsize_, tanh, sigmoid)

end


function update!(ann::ANN, dws::Vector{Array{Float64, 2}},
                 dbs::Vector{Array{Float64, 1}})::ANN
end
    

forward_propagate(input::Array{Float64, 2}, layer::Layer)::Array{Float64, 2} =
    layer.activation(layer.weights * input .+ layer.biases)

forward_propagate(input::Vector{Float64}, layer::Layer)::Vector{Float64} =
    layer.activation(layer.weights * input .+ layer.biases)


function predict(ann::ANN, ip::Vector{Float64})::Vector{Float64}
    a = ip
    for layer = ann.hidden
        a = forward_propagate(a, layer)
    end
    forward_propagate(a, ann.output)
end

function predict(ann::ANN, ip::Array{Float64, 2})::Array{Float64}
    a = ip
    for layer = ann.hidden
        a = forward_propagate(a, layer)
    end
    forward_propagate(a, ann.output)
end


struct ForwPropState
    inputs::Array{Float64, 2}
    hiddenZ::Vector{Array{Float64, 2}}
    hiddenA::Vector{Array{Float64, 2}}
    outZ::Array{Float64, 2}
    outA::Array{Float64, 2}
end

function forward_propagate(ann::ANN,
                           inputs::Array{Float64, 2})::ForwPropState
    Z  = Vector{Array{Float64, 2}}(0)
    A  = Vector{Array{Float64, 2}}(0)
    a  = inputs
    for layer = ann.hidden
        z = layer.weights * a .+ layer.biases
        a = layer.activation.(z)
        push!(Z, z)
        push!(A, a)
    end

    opz = ann.output.weights * a .+ ann.output.biases
    opa = ann.output.activation(opz)

    ForwPropState(inputs, Z, A, opz, opa)
end

struct BackPropUpdate
    outWeights::Array{Float64, 2}
    outBiases::Array{Float64, 1}
    hiddenWeights::Vector{Array{Float64, 2}}
    hiddenBiases::Vector{Array{Float64, 1}}
end

function backward_propagate!(ann::ANN,
                            fp::ForwPropState,
                            outputs::Array{Float64, 2})::BackPropUpdate
    m = size(outputs, 2)

    daop = (fp.outA .- outputs) ./ (fp.outA .* (1.0 .- fp.outA))
    #dzo  = daop .* deriv(ann.output.activation)(fp.outZ)
    dzop = fp.outA .- outputs

    dwop = (dzop * transpose(fp.hiddenA[end])) ./ m
    dbop = (dzop * ones(m)) ./ m

    dwsHidden = Vector{Array{Float64, 2}}(0)
    dbsHidden = Vector{Array{Float64, 1}}(0)

    da = transpose(ann.output.weights) * dzop
    for l=(ann.depth-1):-1:2
        z  = fp.hiddenZ[l]
        dz = da .* tanhDeriv.(z)
        dw = (dz * transpose(fp.hiddenA[l-1])) ./ m
        insert!(dwsHidden, 1, dw)
        db = (dz * ones(m)) ./ m
        insert!(dbsHidden, 1, db)
        da = transpose(ann.hidden[l].weights) * dz
    end
    z = fp.hiddenZ[1]
    dz = da .* tanhDeriv(z)
    dw = (dz * transpose(fp.inputs)) ./ m
    insert!(dwsHidden, 1, dw)
    db = (dz * ones(m)) ./ m
    insert!(dbsHidden, 1, db)

    BackPropUpdate(dwop, dbop, dwsHidden, dbsHidden)
end


function update!(layer::Layer, dW::Array{Float64, 2},
                 db::Array{Float64, 1})::Layer
    layer.weights = layer.weights .- dW
    layer.biases  = layer.biases  .- db
    layer
end

function update!(ann::ANN, bp::BackPropUpdate)::ANN
    update!(ann.output, bp.outWeights, bp.outBiases)
    for l = 1:(ann.size - 1)
        update!(ann.hidden[l], bp.hiddenWeights[l], bp.hiddenBiases[l])
    end
    ann
end


    
