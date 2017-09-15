#an artificial neural network should contain weights for layers,
#and their connectivity

#a neural unit in a layer combines the outputs (activity) of its input units,
#to compute is total (combined) input --
#which then passes through an activation function to become that unit's activity
#While this story is built around individual neural unit,
#in practice we work with the whole layer as a vector unit.
#We also encode connectivity as between layers and not individual units.
#So we extend all of these terms to the whole layer,
#making a layer as the unit of our computations.

LayerID = UInt8
Float2D = Array{Float64, 2}
Float1D = Vector{Float64}
#weights and biases determine the transformation of information
#between layers. In a simple linearly connected network,
#the id of the receiving layer is enough to track the weights.
#in a general setting, we will need the ids of both the out and in layers

struct Arrow
    to::LayerID
    from::LayerID
end

struct Network
    depth::UInt8
    weights::Dict{Arrow, Float2D}
    biases::Dict{LayerID, Float1D}
    actifun::Dict{LayerID, Function}
    #connectivity
    inlayers::Dict{LayerID, Vector{LayerID}}
    outlayers::Dict{LayerID, Vector{LayerID}}
end
weights{WB}(n::WB, a::Arrow) = n.weights[a]
weights{WB}(n::WB, lto::LayerID, lfrom::LayerID) = n.weights[Arrow(lto, lfrom)]
#or get weights assuming a linearly connected network
weights{WB}(n::WB, l::LayerID) = n.weights[Arrow(l, l - 1)]
#we can also have setters
function setw!{WB}(n::WB, a::Arrow, w::Float2D)
    n.weights[a] = w
    w
end
setw!{WB}(n::WB, lt::LayerID, lf::LayerID, w::Float2D) = setw!(n, Arrow(lt, lf), w)
setw!{WB}(n::WB, l::LayerID, w::Float2D) = setw!(n, Arrow(l, l-1), w)

biases{WB}(n::WB, l::LayerID) = n.biases[l]
function setb!{WB}(n::WB, l::LayerID, b::Float1D)
    n.biases[l] = b
    b
end

actifun(ann::Network, a::Arrow) = ann.actifun[a]
actifun(ann::Network, l::LayerID) = ann.actifun[l]

inlayers(ann::Network,  l::LayerID) = ann.inlayers[l]
outlayers(ann::Network, l::LayerID) = ann.outlayers[l]
layers(ann::Network) = (k for (k, v) = ann.inlayers if size(v, 1) != 0)

outputLayer(ann::Network) = ann.depth

#along with Network, we will an object that can contain a network layers' state
struct NState{IOtype}
    number::Int
    activity::Dict{LayerID, IOtype}
    zee::Dict{LayerID, IOtype}
end

negs{IOtype}(ns::NState{IOtype})::Int = ns.number


activity{IOtype}(s::NState{IOtype}, l::LayerID) = s.activity[l]
function seta!{IOtype}(s::NState{IOtype}, l::LayerID, a::IOtype)
    s.activity[l] = a
    a
end
zee{IOtype}(s::NState{IOtype}, l::LayerID) = s.zee[l]
function setip!{IOtype}(s::NState{IOtype}, l::LayerID, ip::IOtype)
    s.zee[l] = ip
    ip
end
setz!{IOtype}(s::NState{IOtype}, l::LayerID, z::IOtype) = setip!(s, l, z)

#to store state gradients
NSGrad{IOtype} = NState{IOtype}

#to compute the zee to a layer, we need activities of its in layers
#we pass known layer activities and zees as a dictionary
#and also provide banged versions of these functions
#that will set the computed value in the dictionaries
activity{IOtype}(n::Network, s::NState{IOtype}, l::LayerID) =
    haskey(s.activity, l) ? s.activity[l] : actifun(n, l)(zee(n, s, l))

function activity!{IOtype}(nn::Network, ns::NState{IOtype}, l::LayerID)
    a = activity(nn, ns, l)
    ns.activity[l] = a
    a
end

zee{IOtype}(n::Network, s::NState{IOtype}, l::LayerID, i::LayerID) =
    weights(n, l, i) * activity(n, s, i) .+ biases(n, l)

zee{IOtype}(n::Network, s::NState{IOtype}, l::LayerID) =
    haskey(s.zee, l) ? s.zee[l] : sum(map(i->zee(n, s, l, i), inlayers(n, l)))

function zee!{IOtype}(n::Network,
                      s::NState{IOtype},
                      l::LayerID)
    ip = zee(n, s, l)
    setip!(s, l, ip)
    ip
end

#for back prop
backact{IOT}(n::Network, s::NState{IOT}, g::NSGrad{IOT}, o::LayerID, l::LayerID) =
    transpose(weights(n, o, l)) * backzee(n, s, g, o) 

backact{IOtype}(n::Network, s::NState{IOtype}, g::NSGrad{IOtype}, l::LayerID) =
    haskey(g.activity, l) ? g.activity[l] : sum(map(o->backact(n, s, g, o, l),
                                                    outlayers(n, l)))
function backact!{IOtype}(n::Network,
                          s::NState,
                          g::NSGrad{IOtype},
                          l::LayerID)
    da = backact(n, s, g, l)
    seta!(g, l, da)
    da
end
                                                
backzee{IOtype}(n::Network, s::NState{IOtype}, g::NSGrad{IOtype}, l::LayerID) =
    haskey(g.zee, l) ? g.zee[l] : backact(n, s, g, l) .* deriv(actifun(n, l))(zee(n, s, l))

function backzee!{IOtype}(n::Network,
                          s::NState{IOtype},
                          g::NSGrad{IOtype},
                          l::LayerID)
    dz = backzee(n, s, g, l)
    setz!(g, l, dz)
    dz
end

#iterators
#to order the layers, we need to determine ancestorship
isanc(n::Network, x::LayerID, y::LayerID)::Bool = y in n.inlayers[x]
isdsc(n::Network, x::LayerID, y::LayerID)::Bool = y in n.outlayers[x]
    
function place!(ann::Network, l::LayerID, layers::Vector{LayerID})
    insertAfterIf!(layers, l, x->isanc(ann, l, x))
end

function forwItrLayers(ann::Network)
    layers = [0x00]
    for l = 1:ann.depth
        place!(ann, UInt8(l), layers)
    end
    layers[2:end]
end

#the last element in forward iterated layers should be the output layer
forwItrHidden(ann::Network) = forwItrLayers(ann)[1:end - 1]

backItrHidden(ann::Network) = reverse(forwItrHidden(ann))
backItrLayers(ann::Network) = reverse(forwItrLayers(ann))

struct NWGrad
    weights::Dict{Arrow, Float2D}
    biases::Dict{LayerID, Float1D}
end

wgrad{IOT}(ns::NState{IOT}, ng::NSGrad{IOT}, l::LayerID, i::LayerID) =
    (ng.zee[l] * transpose(ns.activity[i])) ./ negs(ns)

function bgrad{IOT}(ng::NSGrad{IOT}, l::LayerID)::Vector{Float64}
    z = ng.zee[l]
    n = size(z, 1)
    reshape(sum(ng.zee[l], 2), (n, )) ./ negs(ng)
end

function wbgrad{IOT}(ann::Network, ns::NState{IOT}, nsg::NSGrad{IOT})::NWGrad
    m = negs(ns)
    nwg = NWGrad(Dict{Arrow, Float2D}(), Dict{LayerID, Float1D}())
    for l = layers(ann)
        for i = inlayers(ann, l)
            setw!(nwg, l, i, wgrad(ns, nsg, l, i))
        end
        setb!(nwg, l, bgrad(nsg, l))
    end
    nwg
end

function update!(n::Network, nwg::NWGrad, alpha::Float64)::Network
    for l = layers(n)
        for i = inlayers(n, l)
            n.weights[Arrow(l, i)] -= alpha * nwg.weights[Arrow(l, i)]
        end
        n.biases[l] -= alpha * nwg.biases[l]
    end
    n
end


    
