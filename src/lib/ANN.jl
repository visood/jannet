module ANN

include("annutils.jl")
export cons, sigmoid, Actifun, deriv, findIf, insertAfterIf!

include("network.jl")
export LayerID, Float2D, Float1D,
    Arrow, Network, weights, setw!, biases, setb!, actifun,
    inlayers, outlayers, layers,
    NState, activity, seta!, zee, setip!, setz!, NSGrad, activity,
    zee!, backact, backact!, backzee, backzee!,
    forwItrHidden, forwItrLayers, backItrHidden, backItrLayers,
    place!, isanc, isdsc,
    wbgrad, wgrad, bgrad, update!

include("propagate.jl")
export forwprop, backprop

include("train.jl")
export train!
end
