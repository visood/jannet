function forwprop{IOtype}(ann::Network, input::IOtype)::NState{IOtype}
    ns = NState(size(input, 2),
                Dict{LayerID, IOtype}(),
                Dict{LayerID, IOtype}())
    seta!(ns, 0x00, input)
    for l = forwItrHidden(ann)
        z = zee(ann, ns, l)
        setip!(ns, l, z)
        a = actifun(ann, l).(z)
        seta!(ns, l, a)
    end
    ol = outputLayer(ann)
    z = zee(ann, ns, ol)
    setz!(ns, ol, z)
    a = actifun(ann, ol).(z)
    seta!(ns, ol, a)
    ns
end

function backprop{IOtype}(ann::Network,
                          ns::NState{IOtype},
                          target::IOtype)::NSGrad{IOtype}
    nsg = NSGrad(size(target, 1),
                 Dict{LayerID, IOtype}(),
                 Dict{LayerID, IOtype}())
    ol   = outputLayer(ann)
    z    = zee(ns, ol)
    dz   = activity(ns, ol) - target
    setz!(nsg, ol, dz)
    da   = dz ./ deriv(actifun(ann, ol)).(z)
    seta!(nsg, ol, da)
    seta!(nsg, ol, da)

    for l = backItrHidden(ann)
        da = backact(ann, ns, nsg, l)
        seta!(nsg, l, da)
        z  = zee(ann, ns, l)
        dz = da .* deriv(actifun(ann, l)).(z)
        setz!(nsg, l, dz)
    end

    nsg
end
