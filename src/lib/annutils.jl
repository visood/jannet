#utils for ANNs
cons{T}(x::T, ys::Array{T, 1})::Array{T, 1} = insert!(copy(ys), 1, x)#prepend!(copy(ys), [x])
cons{T}(ys::Array{T, 1}, x::T)::Array{T, 1} = push!(copy(ys), x)
    
struct Actifun
    at::Function
    deriv::Function
end

sigmoid{T}(Z::T)::T = 1.0 ./ (1.0 .+ exp.(-Z))
sigmoidDeriv{T}(Z::T)::T = sigmoid(Z) .* (1.0 .- sigmoid(Z))
sigmoidActifun = Actifun(sigmoid, sigmoidDeriv)

tanhDeriv{T}(Z::T)::T = 1.0 .- tanh.(Z) .^ 2
tanhActifun = Actifun(tanh, tanhDeriv)

identityActifun = Actifun(z -> z, z -> ones(size(z)))
                                
function deriv(f::Function)::Function
    dict = Dict(sigmoid => sigmoidDeriv,
                tanh    => tanhDeriv)
    haskey(dict, f) ? dict[f] :
        throw(ArgumentError("derivative for %f not specified"))
end

function findIf{T}(xs::Vector{T}, pred::Function)
    i = 1
    for x = xs
        if pred(x) break
        else i += 1
        end
    end
    i
end

function insertAfterIf!{T}(xs::Vector{T}, y::T, pred::Function)
    i = findIf(xs, pred)
    insert!(xs, i + 1, y)
end
