function train!(ann::Network,
                inputs::Float2D,
                targets::Float2D,
                numIterations::Int,
                alpha::Float64)::Network
    for i = 1:numIterations
        ns  = forwprop(ann, inputs)
        nsg = backprop(ann, ns, targets)
        wbg  = wbgrad(ann, ns, nsg)
        update!(ann, wbg, alpha)
    end
    ann
end

